lr=0.1
CUDA_VISIBLE_DEVICES=0 \
python main_moco.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 4096 \
  --cos --mlp \
  --moco-t 0.2 \
  --amp-opt-level O1 \
  --knn-k 20 \
  --knn-t 0.02 \
  --knn-data PATH \
  --save-dir "output/kuzikus/mocov2/moco-v2-epochs200-LT/" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  PRETRAIN_PATH
