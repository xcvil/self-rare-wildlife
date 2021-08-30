lr=0.1
Lambda=0.25
cld_t=0.4
clusters=32
CUDA_VISIBLE_DEVICES=0,1 \
python main_moco_cld.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 4096 \
  --Lambda ${Lambda} \
  --aug-color --cos --mlp \
  --moco-t 0.2 \
  --cld-t ${cld_t} \
  --amp-opt-level O1 \
  --num-iters 16 \
  --clusters ${clusters} \
  --use-kmeans \
  --normlinear \
  --knn-k 20 \
  --knn-t 0.02 \
  --knn-data PATH \
  --save-dir "output/imagenet/mocov2+cld/lr${lr}-Lambda${Lambda}-cld_t${cld_t}-clusters${clusters}-NormNLP-epochs200/" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  PRETRAIN_PATH
