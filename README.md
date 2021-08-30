# Self-Supervised Pretraining and Controlled Augmentation Improve Rare Wildlife Recognition in UAV Images

This is a PyTorch implementation of the [Self-Supervised Pretraining and Controlled Augmentation Improve Rare Wildlife Recognition in UAV Images](https://arxiv.org/abs/2108.07582):
```
@Article{zheng2021selfkuzikus,
  title = {Self-Supervised Pretraining and Controlled Augmentation Improve Rare Wildlife Recognition in UAV Images},
  author = {Zheng, Xiaochen and Kellenberger, Benjamin and Gong, Rui and Hajnsek, Irena and Tuia, Devis},
  journal = {arXiv preprint arXiv:2108.07582},
  year = {2021}
}
```

### Updates
[30/08/2021] Feature extraction (for t-SNE visualization and KNN grid search) is supported, see [here](cld/vis.py).

[26/08/2021] Training MoCo + CLD with domain-specific geometric augmentation (GeoCLD) is supported, see [here](cld/pretrain_cld_geo_color_shared_head.sh).

### Requirements
* Python >= 3.7, < 3.9
* PyTorch >= 1.6
* pandas
* NumPy
* tqdm
* [apex](https://github.com/NVIDIA/apex) (optional, unless using mixed precision training)

### Tools
#### Feature Extraction (Encoder+MLP Projection, Instance Discrimination Branch)
```shell
python vis.py \
  -a resnet50 \
  --resume [YOUR_PTH_TAR_MODEL_FILE] \
  --save-dir [SAVE_DIR] \
  --mlp \
  --moco-k 4096 
```
#### KNN Parameters Grid Search
```shell
python vis.py \
  -a resnet50 \
  --resume [YOUR_PTH_TAR_MODEL_FILE] \
  --save-dir [SAVE_DIR] \
  --mlp \
  --moco-k 4096 \
  --knn-search \
  --knn-k k \
  --knn-t t \
  --knn-data [DATASET_FOLDER_WITH_TRAIN_VAL_FOLDERS]
```


### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details. 

### Acknowledgements
The authors would like to thank the [Kuzikus Wildlife Reserve](https://www.kuzikus-namibia.de), Namibia for the access to the aerial data and the ground reference used in this study.

Part of this code is based on [MoCo](https://github.com/facebookresearch/moco), [CLD](https://github.com/frank-xwang/CLD-UnsupervisedLearning), [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup), and [CIFAR demo on Colab GPU](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb).