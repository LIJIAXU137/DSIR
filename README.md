# DSIR

  python main.py DSIR2 --dataset ham10000 --base-ratio 0.42 --phases 4
 --data-root ~/dataset --batch-size 32 --num-workers 16 --backbone resnet32  --learning-rate 0.5 --label-smoothing 0 --base-epochs 100 --gamma 0.1 --gamma-comp 0.1 --compensation-ratio 0.6 --buffer-size 1024 --cache-features --IL-batch-size 4096 --gpu 1

