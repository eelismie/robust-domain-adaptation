CUDA_VISIBLE_DEVICES=0 python mcc.py data/digits -d Digits -s SVHNRGB -t MNISTRGB --train-resizing 'res.' --val-resizing 'res.' --resize-size 32 --no-hflip --norm-mean 0.5 0.5 0.5 --norm-std 0.5 0.5 0.5 -a dtn --no-pool --lr 0.01 -b 128 -i 2500 --scratch --seed 0 --log logs/mcc/SVHN2MNIST --per-class-eval --epochs 20

