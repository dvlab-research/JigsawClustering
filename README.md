# JigsawClustering
**Jigsaw Clustering for Unsupervised Visual Representation Learning**

Pengguang Chen, Shu Liu, Jiaya Jia

## Introduction
This project provides an implementation for the CVPR 2021 paper "[Jigsaw Clustering for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2104.00323.pdf)"

## Installation

### Environment

We verify our code on 
* 4x2080Ti GPUs
* CUDA 10.1
* python 3.7
* torch 1.6.0
* torchvision 0.7.0

Other similar envirouments should also work properly.

### Install

We use the SyncBN from apex, please install apex refer to https://github.com/NVIDIA/apex (SyncBN from pytorch should also work properly, we will verify it later.)

We use detectron2 for the training of detection tasks. If you are willing to finetune our pretrained model on the 
detection task, please install detectron2 refer to https://github.com/facebookresearch/detectron2

```
git clone https://github.com/Jia-Research-Lab/JigsawClustering.git
cd JigsawClustering/
pip install diffdist
```

### Dataset

Please put the data under ./datasets. The directory looks like:

```
datasets
│
│───ImageNet/
│   │───class1/
│   │───class2/
│   │   ...
│   └───class1000/
│   
│───coco/
│   │───annotations/
│   │───train2017/
│   └───val2017/
│
│───VOC2012/
│   
└───VOC2007/
```

## Results and pretrained model
The pretrained model is available at [here](https://github.com/Jia-Research-Lab/JigsawClustering/releases/download/1.0/JigClu_200e.pth).

| Task                | Dataset  | Results |
|---------------------|----------|---------|
| Linear Evaluation   | ImageNet | 66.4    |
| Semi-Supervised 1%  | ImageNet | 40.7    |
| Semi-Supervised 10% | ImageNet | 63.0    |
| Detection           | COCO     | 39.3    |



## Training

### Pre-training on ImageNet

```
python main.py --dist-url 'tcp://localhost:10107' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a resnet50 \
    --lr 0.03 --batch-size 256 --epoch 200 \
    --save-dir outputs/jigclu_pretrain/ \
    --resume outputs/jigclu_pretrain/model_best.pth.tar \
    --loss-t 0.3 \
    --cross-ratio 0.3 \
    datasets/ImageNet/
```
### Linear evaluation on ImageNet

```
python main_lincls.py --dist-url 'tcp://localhost:10007' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a resnet50 \
    --lr 10.0 --batch-size 256 \
    --prefix module.encoder. \
    --pretrained outputs/jigclu_pretrain/model_best.pth.tar \
    --save-dir outputs/jigclu_linear/ \
    datasets/ImageNet/
```

### Semi-Supervised finetune on ImageNet

10% label
```
python main_semi.py --dist-url 'tcp://localhost:10102' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a resnet50 \
    --batch-size 256 \
    --wd 0.0 --lr 0.01 --lr-last-layer 0.2 \
    --syncbn \
    --prefix module.encoder. \
    --labels-perc 10 \
    --pretrained outputs/jigclu_pretrain/model_best.pth.tar \
    --save-dir outputs/jigclu_semi_10p/ \
    datasets/ImageNet/
```

1% label
```
python main_semi.py --dist-url 'tcp://localhost:10101' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a resnet50 \
    --batch-size 256 \
    --wd 0.0 --lr 0.02 --lr-last-layer 5.0 \
    --syncbn \
    --prefix module.encoder. \
    --labels-perc 1 \
    --pretrained outputs/jigclu_pretrain/model_best.pth.tar \
    --save-dir outputs/jigclu_semi_1p/ \
    datasets/ImageNet/
```

### Transfer to COCO detection

Please convert the pretrained weight first
```
python detection/convert.py
```

Then start training using
```
python detection/train_net.py --config-file detection/configs/R50-JigClu.yaml --num-gpus 4
```

VOC detection
```
python detection/train_net.py --config-file detection/configs/voc-R50-JigClu.yaml --num-gpus 4
```

## <a name="Citation"></a>Citation

Please consider citing JigsawClustering in your publications if it helps your research.

```bib
@inproceedings{chen2021jigclu,
    title={Jigsaw Clustering for Unsupervised Visual Representation Learning},
    author={Pengguang Chen, Shu Liu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021},
}
```
