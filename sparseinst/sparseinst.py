# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import einops
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .encoder import build_sparse_inst_encoder
from .decoder import build_sparse_inst_decoder
from .loss import build_sparse_inst_criterion
from .utils import nested_tensor_from_tensor_list
from .util import box_ops

from mask2former_video.utils.memory import retry_if_cuda_oom
from scipy.optimize import linear_sum_assignment

__all__ = ["SparseInst"]


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


@META_ARCH_REGISTRY.register()
class SparseInst(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # move to target device
        self.device = torch.device(cfg.MODEL.DEVICE)

        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        output_shape = self.backbone.output_shape()

        # encoder & decoder
        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)
        # data and preprocessing
        self.mask_format = cfg.INPUT.MASK_FORMAT
        
        # matcher & loss (matcher is built in loss)
        self.criterion = build_sparse_inst_criterion(cfg)

        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # inference
        self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
        
        # Video Task
        self.video_task = cfg.MODEL.SPARSE_INST.VIDEO_TASK
        if self.video_task:
            self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
            self.window_inference = cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE
            self.use_contrastive = cfg.INPUT.CONTRASTIVE_LEARNING
            self.video_info = {'bz':cfg.DATALOADER.DATASET_BS[-1], 'len':cfg.INPUT.SAMPLING_FRAME_NUM}

    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            gt_classes = targets_per_image.gt_classes
            target["labels"] = gt_classes.to(self.device)
            h, w = targets_per_image.image_size
            if not targets_per_image.has('gt_masks'):
                gt_masks = BitMasks(torch.empty(0, h, w))
            else:
                gt_masks = targets_per_image.gt_masks
                if self.mask_format == "polygon":
                    if len(gt_masks.polygons) == 0:
                        gt_masks = BitMasks(torch.empty(0, h, w))
                    else:
                        gt_masks = BitMasks.from_polygon_masks(gt_masks.polygons, h, w)

            target["masks"] = gt_masks.to(self.device)
            new_targets.append(target)

        return new_targets

    def prepare_targets_video(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances
    
    def prepare_targets_video_contras(self, batched_inputs, targets, images):
        img_long_size = max(images.tensor.shape[-2:])
           
        video_targets  = []
        for batched_inputs_i, targets_i in zip(batched_inputs, targets):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            
            for frame_i, targets_per_image in enumerate(targets_i):
                targets_per_image = targets_per_image.to(self.device)
                if  'gt_masks' not in targets_per_image._fields.keys():
                    padded_masks = None
                else:
                    # gt_masks = targets_per_image.gt_masks.tensor
                    if isinstance(targets_per_image.gt_masks, torch.Tensor):
                        gt_masks = targets_per_image.gt_masks
                    else:
                        gt_masks = targets_per_image.gt_masks.tensor
                    padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                    padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                gt_classes = targets_per_image.gt_classes
                
                image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
                gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                gt_boxes = torch.clamp(gt_boxes,0,1)

                inst_ids = targets_per_image.gt_ids

                valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
                if 'ori_id' in targets_per_image._fields.keys():
                    ori_id = [int(oriid) for oriid in targets_per_image.ori_id]
                else:
                    ori_id = None

                if padded_masks is None:
                    video_targets.append(
                    {
                        "labels": gt_classes[valid_id],
                        'inst_id':inst_ids[valid_id],
                        "masks": None,
                        "boxes":gt_boxes[valid_id],
                        "ori_id":ori_id,
                    }
                    )
                else:
                    video_targets.append(
                        {
                            "labels": gt_classes[valid_id],
                            'inst_id':inst_ids[valid_id],
                            "masks": BitMasks(padded_masks[valid_id]),
                            "boxes":gt_boxes[valid_id],
                            "ori_id":ori_id,
                        }
                    )
    
            return video_targets

    def frame_decoder_loss_reshape(self, outputs, targets):
        # Prepare for targets
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        outputs['pred_scores'] = einops.rearrange(outputs['pred_scores'], 'b t q c -> (b t) q c')

        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                # Remove time dimension & Tensor masks -> BitMasks
                masks = BitMasks(masks.flatten(0, 1))
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
                
        # Remove time dimension
        outputs['pred_masks'] = outputs['pred_masks'].squeeze(2)

        return outputs, gt_instances

    def forward(self, batched_inputs):
        if self.video_task:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(frame.to(self.device))
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, 32)
        else:
            images = self.preprocess_inputs(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        max_shape = images.tensor.shape[2:]
        # forward
        if not self.training and self.video_task and self.window_inference:
            outputs = self.run_window_inference(images.tensor)
        else:
            features = self.backbone(images.tensor)
            features = self.encoder(features)
            outputs = self.decoder(features)
        if self.training:
            if self.video_task:
                if self.use_contrastive:
                    gt_instances = [x["instances"] for x in batched_inputs]
                    targets = self.prepare_targets_video_contras(batched_inputs, gt_instances, images)
                else:
                    targets = self.prepare_targets_video(batched_inputs, images)
                    outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)
                losses = self.criterion(outputs, targets, max_shape)
                if self.use_contrastive:
                    track_loss = self.get_tracking_contrastive_lossv3(outputs, targets, max_shape)
                    losses.update({"track_loss": track_loss})
                    losses["track_loss"] *= 2 # track_loss weight
            else:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances)
                losses = self.criterion(outputs, targets, max_shape)
            return losses
        else:
            if self.video_task:
                outputs = self.post_processing(outputs)
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_cls_result = mask_cls_results[0]   # t, q, C -> q, C
                mask_pred_result = mask_pred_results[0]
                first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

                input_per_image = batched_inputs[0]
                image_size = images.image_sizes[0]  # image size without padding after data augmentation

                height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
                width = input_per_image.get("width", image_size[1])
                return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size)
            else:
                results = self.inference(outputs, batched_inputs, max_shape, images.image_sizes)
                processed_results = [{"instances": r} for r in results]
            return processed_results

    def forward_test(self, images):
        # for inference, onnx, tensorrt
        # input images: BxCxHxW, fixed, need padding size
        # normalize
        images = (images - self.pixel_mean[None]) / self.pixel_std[None]
        features = self.backbone(images)
        features = self.encoder(features)
        output = self.decoder(features)

        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)
        pred_masks = F.interpolate(
            pred_masks, scale_factor=4.0, mode="bilinear", align_corners=False)
        return pred_scores, pred_masks

    def inference(self, output, batched_inputs, max_shape, image_sizes):
        # max_detections = self.max_detections
        results = []
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        for _, (scores_per_image, mask_pred_per_image, batched_input, img_shape) in enumerate(zip(
                pred_scores, pred_masks, batched_inputs, image_sizes)):

            ori_shape = (batched_input["height"], batched_input["width"])
            result = Instances(ori_shape)
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.pred_classes = labels
                results.append(result)
                continue

            h, w = img_shape
            # rescoring mask using maskness
            scores = rescoring_mask(scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image)

            # upsample the masks to the original resolution:
            # (1) upsampling the masks to the padded inputs, remove the padding area
            # (2) upsampling/downsampling the masks to the original sizes
            mask_pred_per_image = F.interpolate(mask_pred_per_image.unsqueeze(1), size=max_shape, mode="bilinear", align_corners=False)[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(mask_pred_per_image, size=ori_shape, mode='bilinear', align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            # fix the bug for visualization
            # mask_pred = BitMasks(mask_pred)

            # using Detectron2 Instances to store the final results
            result.pred_masks = mask_pred
            result.scores = scores
            result.pred_classes = labels
            results.append(result)

        return results

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size
            features = self.backbone(images_tensor[start_idx:end_idx])
            features = self.encoder(features)
            out = self.decoder(features)
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_scores'] = torch.cat([x['pred_scores'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing(self, outputs):
        pred_logits, pred_scores, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_scores'], outputs['pred_masks'], outputs['pred_embds']
        # pred_logits = pred_logits.sigmoid()
        # pred_objectness = pred_scores.sigmoid()
        # pred_logits = torch.sqrt(pred_logits * pred_objectness)
        pred_logits = pred_logits.softmax(dim=-1)
        pred_masks = pred_masks.sigmoid()
        
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        for i in range(1, len(pred_logits)):
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)
        
        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)
            nq = pred_cls.shape[0]
            labels = torch.arange(self.decoder.inst_branch.num_classes, device=self.device).unsqueeze(0).repeat(nq, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.decoder.inst_branch.num_classes
            pred_masks = pred_masks[topk_indices]
            # import pdb; pdb.set_trace()
            # scores = rescoring_mask(scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image)
            
            pred_masks = F.interpolate(pred_masks, size=first_resize_size, mode="bilinear", align_corners=False)

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False)

            masks = pred_masks > 0

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
    
    def get_tracking_contrastive_lossv3(self, video_outputs, video_targets, input_shape):  # IDOL track loss
        indices_all = self.criterion.matcher(video_outputs, video_targets, input_shape)
        
        video_len = self.video_info['len']
        track_loss = 0
        num_inst = 0
        batch_similarity = []
        batch_label = []

        for i in range(self.video_info['bz']): # 每个batch 切片操作
            indices = indices_all[i*video_len:(i+1)*video_len]
            bz_embedding = video_outputs['pred_embds'][i*video_len:(i+1)*video_len]
            bz_target = video_targets[i*video_len:(i+1)*video_len]
            zero = torch.tensor(0).to(bz_embedding.device)
            one = torch.tensor(1).to(bz_embedding.device)
            video_contras = {}
            memory = {}
            for f,(findice,fembed,ftarget) in enumerate(zip(indices,bz_embedding,bz_target)):
                vf_embed_k = fembed[findice[0].long()]
                if len(vf_embed_k.shape) ==1:
                    vf_embed_k.unsqueeze(0)
                vf_gt_id_k = ftarget['inst_id'][findice[1].long()]

                # neg sample
                sampled_index = set(random.sample(range(100), 20)) 
                neg_index = sampled_index - set(findice[0].tolist())
                neg_index = list(neg_index)
                vf_embed_neg = fembed[neg_index]
                vf_embed = torch.cat([vf_embed_k,vf_embed_neg],dim=0)
                vf_gt_id = torch.cat([vf_gt_id_k,zero.repeat(len(neg_index))-2],dim=0) 

                video_contras[f] = (vf_embed,vf_gt_id)

                if f > 0:
                    num_inst = num_inst + len(ftarget['inst_id'])
                    similarity_matric =  torch.einsum("ac,bc->ab", video_contras[f-1][0], vf_embed_k)  #[num_1, num_gt]

                    v0_gt_id_m = video_contras[f-1][1].unsqueeze(-1).repeat(1,len(vf_gt_id_k))
                    v1_gt_id_m = vf_gt_id_k.unsqueeze(0).repeat(len(video_contras[f-1][1]),1)
                    similarity_label = (v0_gt_id_m == v1_gt_id_m).float()  # can be treat as one hot label 
                    # use focal loss instand of contrastive
                    # aux  cosine
                    # aux_contrastive_embed=nn.functional.normalize(video_contras[f-1][0].float(),dim=1)
                    # key_embed_i=nn.functional.normalize(vf_embed_k.float(),dim=1)    
                    # cosine = torch.einsum('nc,kc->nk',[aux_contrastive_embed,key_embed_i])

                    # batch_similarity_aux.append(cosine.flatten() )
                    batch_similarity.append(similarity_matric.flatten() )
                    batch_label.append(similarity_label.flatten() )
        if len(batch_similarity)==0 or torch.cat(batch_similarity).shape[0] == 0:
            track_loss = (video_outputs['pred_embds']*0).sum()
        else:
            contras_loss = 0
            aux_loss = 0
            for pred, label in zip(batch_similarity, batch_label):
                if len(pred) == 0:
                    continue
                pred = pred.unsqueeze(0)
                label = label.unsqueeze(0)
                # aux_pred = aux_pred.unsqueeze(0)

                pos_inds = (label == 1)
                neg_inds = (label == 0)
                pred_pos = pred * pos_inds.float()
                pred_neg = pred * neg_inds.float()
                # use -inf to mask out unwanted elements.
                pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
                pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')
                _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
                _neg_expand = pred_neg.repeat(1, pred.shape[1])
                # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
                x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0) 
                contras_loss += torch.logsumexp(x, dim=1)

            # track_loss = (contras_loss + 1.5*aux_loss)
            track_loss = contras_loss/max(num_inst,1)

        track_loss = track_loss #  /(self.video_info['bz'])
        return track_loss


