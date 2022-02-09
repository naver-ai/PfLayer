## Learning Features with Parameter-Free Layers (ICLR 2022)

**Dongyoon Han, YoungJoon Yoo, Beomyoung Kim, Byeongho Heo** | [Paper](https://arxiv.org/pdf/2202.02777.pdf)

**NAVER AI Lab, NAVER CLOVA** 


### Updates
- **02.09.2002** Code will be uploaded soon (due to internal code review)
- **02.06.2022** Initial update 

### Abstract

Trainable layers such as convolutional building blocks are the standard network design choices by learning parameters to capture the global context through successive spatial operations. When designing an efficient network, trainable layers such as the depthwise convolution is the source of efficiency in the number of parameters and FLOPs, but there was little improvement to the model speed in practice. This paper argues that **simple built-in parameter-free operations can be a favorable alternative to the efficient trainable layers** replacing spatial operations in a network architecture. We aim to **break the stereotype of organizing the spatial operations of building blocks into trainable layers.** 
Extensive experimental analyses based on layer-level studies with fully-trained models and neural architecture searches are provided to investigate whether parameter-free operations such as the max-pool are functional. The studies eventually give us **a simple yet effective idea for redesigning network architectures**, where the parameter-free operations are heavily used as the main building block without sacrificing the model accuracy as much. Experimental results on the ImageNet dataset demonstrate that the network architectures with parameter-free operations could enjoy the advantages of further efficiency in terms of model speed, the number of the parameters, and FLOPs.

### Some Analyses in The Paper
#### 1. Depthwise convolution is replaceble with a parameter-free operation:
<img src=https://user-images.githubusercontent.com/31481676/152579508-77401a51-7c86-401a-a050-b2033e2e9498.png width=760>

#### 2. Parameter-free operations are frequently searched in normal building blocks by NAS:
<img src=https://user-images.githubusercontent.com/31481676/152579742-be646019-128d-4771-9d00-ad5a645c128b.png width=760>

#### 3. R50-hybrid (with eff-bottlenecks) yields a localizable features (see the Grad-CAM visualizations):
<img src=https://user-images.githubusercontent.com/31481676/152579400-05b95b4b-a915-4f38-8639-4b0f4080c532.png width=840>


### Our Proposed Models
#### 1. Schematic illustration of our models
<img src=https://user-images.githubusercontent.com/31481676/152576171-7fa44cbf-ccfc-4414-9ff1-32e9e6b85799.png width=720>

- Here, we provide example models where the parameter-free operations (i.e., eff-layer) are mainly used;

- Parameter-free operations such as the max-pool2d and avg-pool2d can replace the spatial operations (conv and SA).


#### 2. Brief model descriptions
   ``resnet_pf.py: resnet50_max(), resnet50_hybrid()``: R50-max, R50-hybrid - model with the efficient bottlenecks
   
   ``vit_pf.py: vit_max_s()`` - [ViT](https://arxiv.org/abs/2010.11929) with the efficient transformers 
   
   ``pit_pf.py: pit_max_s()`` - [PiT](https://arxiv.org/abs/2103.16302) with the efficient transformers


### Usage
#### Requirements
    pytorch >= 1.6.0
    torchvision >= 0.7.0
    timm >= 0.3.4
    apex == 0.1.0

#### Pretrained models

Network | Img_size | Params. (M) | FLOPs (G) | GPU (ms) | Top-1 (%) | Top-5 (%)
-- | :--: | :--:  | :--: | :--:  | :--: | :--: 
`R50`            | 224x224 | 25.6 | 4.1 | 8.7 | 76.2 | 93.8
[`R50-max`]()    | 224x224 | 14.2 | 2.2 | 6.8 | 74.3 | 92.0  
[`R50-hybrid`]() | 224x224 | 17.3 | 2.6 | 7.3 | 77.1 | 93.1 


Network | Img Size | Throughputs | Vanilla | +CutMix | +Deit
-- | :--: | :--:  | :--: | :--:  | :--:
`R50`          | 224x224 | 962 / **112** | **76.2** | 77.6 | 78.8
[`ViT-S-max`]()    |224x224 |  763 / 96 | 74.2 | 77.3 | 79.8
[`PiT-S-max`]()     |224x224 |  **1000** / 92 |75.7 | **78.1** | **80.1**


#### Training 

- Our **ResNet-based models** can be trained with any PyTorch training codes; we recommend [timm](https://github.com/rwightman/pytorch-image-models). We provide a sample script for training R50_hybrid with the standard 90-epochs training setup:


      python3 -m torch.distributed.launch --nproc_per_node=4 train.py ./ImageNet_dataset/ --model resnet50_hybrid --opt sgd --amp \
      --lr 0.2 --weight-decay 1e-4 --batch-size 256 --sched step --epochs 90 --decay-epochs 30 --warmup-epochs 3 --smoothing 0\


- **Vision transformers (ViT and PiT) models** are also able to be trained with [timm](https://github.com/rwightman/pytorch-image-models), but we recommend the code [DeiT](https://github.com/facebookresearch/deit) to train with.  We provide a sample training script with the default training setup in the package:


      python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_max_s --batch-size 256 --data-path ./ImageNet_dataset/





### How to cite

```
@inproceedings{han2022learning,
    title={Learning Features with Parameter-Free Layers},
    author={Dongyoon Han, YoungJoon Yoo, Beomyoung Kim, Byeongho Heo},
    year={2022},
    journal={International Conference on Learning Representations},
}
```














