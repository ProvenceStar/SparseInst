CUDA_VISIBLE_DEVICES=5 python train_net.py --config-file configs/image/sparse_inst_r50_giam.yaml MODEL.WEIGHTS checkpoints/sparse_inst_r50_giam_aug_2b7d68.pth SOLVER.IMS_PER_BATCH 2