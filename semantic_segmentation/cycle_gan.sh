CUDA_VISIBLE_DEVICES=0 python cycle_gan.py data/Cityscapes data/Cityscapes -s Cityscapes -t FoggyCityscapes \
    --log logs/cyclegan/cityscapes2foggy --translated-root data/Cityscapes2Foggy/CycleGAN_39 --epochs 1
