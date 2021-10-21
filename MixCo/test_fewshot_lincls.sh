CUDA_ID=0
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python main_lincls.py \
  -a resnet50 \
  --lr 30 \
  --worker 2 \
  --batch-size 256 \
  --epochs 100 \
  --finetune-model WEIGHTS_PATH \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  DATASET_PATH
