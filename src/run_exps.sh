#!/bin/bash -l

repo_name=monotone-GANs
dropbox_path=GTDropbox:alisk/$repo_name

if [ -d $HOME/$repo_name/ ]; then
    src_dir=$HOME/$repo_name/
else
    src_dir=$HOME/Codes/$repo_name/
fi

cd $src_dir

d1=mnist_wd-1e-4_batchsize-128_max-epoch-300_lr-2e-4
python src/monotone-gan.py  --example mnist --experiment $d1  \
    --wd 0.0001 --batch_size 128 --max_epoch 300 --lr 0.0002 > $d1.log

d2=mnist_wd-0_batchsize-128_max-epoch-300_lr-2e-4
python src/monotone-gan.py  --example mnist --experiment $d2  \
    --wd 0.0 --batch_size 128 --max_epoch 300 --lr 0.0002 > $d2.log