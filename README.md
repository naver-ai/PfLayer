## Learning Features with Parameter-Free Layers (ICLR 2022)

**Dongyoon Han, YoungJoon Yoo, Beomyoung Kim, Byeongho Heo** | [Paper](https://arxiv.org/pdf/2202.02777.pdf)

**NAVER AI Lab, NAVER CLOVA** 


### Updates
- **07.06.2022** Improved ResNet50 pretrained on ImageNet-1k are added
- **03.23.2022** Performance of ViT and PiT has been updated
- **02.11.2022** Code has been uploaded
- **02.06.2022** Initial update 

### Abstract

Trainable layers such as convolutional building blocks are the standard network design choices by learning parameters to capture the global context through successive spatial operations. When designing an efficient network, trainable layers such as the depthwise convolution is the source of efficiency in the number of parameters and FLOPs, but there was little improvement to the model speed in practice. This paper argues that **simple built-in parameter-free operations can be a favorable alternative to the efficient trainable layers** replacing spatial operations in a network architecture. We aim to **break the stereotype of organizing the spatial operations of building blocks into trainable layers.** 
Extensive experimental analyses based on layer-level studies with fully-trained models and neural architecture searches are provided to investigate whether parameter-free operations such as the max-pool are functional. The studies eventually give us **a simple yet effective idea for redesigning network architectures**, where the parameter-free operations are heavily used as the main building block without sacrificing the model accuracy as much. Experimental results on the ImageNet dataset demonstrate that the network architectures with parameter-free operations could enjoy the advantages of further efficiency in terms of model speed, the number of the parameters, and FLOPs.

### Some Analyses in The Paper
#### 1. Depthwise convolution is replaceble with a parameter-free operation:
<img src=https://user-images.githubusercontent.com/31481676/152579508-77401a51-7c86-401a-a050-b2033e2e9498.png width=760>

#### 2. Parameter-free operations are frequently searched in normal building blocks by NAS (when searching with individual cells):
<img src=https://user-images.githubusercontent.com/31481676/152579742-be646019-128d-4771-9d00-ad5a645c128b.png width=760>

#### 3. R50-hybrid (with the eff-bottlenecks) yields a localizable features (see the Grad-CAM visualizations):
<img src=https://user-images.githubusercontent.com/31481676/152579400-05b95b4b-a915-4f38-8639-4b0f4080c532.png width=840>


### Our Proposed Models
#### 1. Schematic illustration of our models
<img src=https://user-images.githubusercontent.com/31481676/152576171-7fa44cbf-ccfc-4414-9ff1-32e9e6b85799.png width=720>

- Here, we provide example models where the parameter-free operations (i.e., eff-layer) are mainly used;

- Parameter-free operations such as the max-pool2d and avg-pool2d can replace the spatial operations (conv and SA).


#### 2. Brief model descriptions
   ``resnet_pf.py: resnet50_max(), resnet50_hybrid()``: R50-max, R50-hybrid - model with the efficient bottlenecks
   
   ``vit_pf.py: vit_s_max()`` - [ViT](https://arxiv.org/abs/2010.11929) with the efficient transformers 
   
   ``pit_pf.py: pit_s_max()`` - [PiT](https://arxiv.org/abs/2103.16302) with the efficient transformers


### Usage
#### Requirements
    pytorch >= 1.6.0
    torchvision >= 0.7.0
    timm >= 0.3.4
    apex == 0.1.0

#### Pretrained models
- `+` denotes the models trained for longer epochs

Network | Img size | Params. (M) | FLOPs (G) | GPU (ms) | Top-1 (%) | Top-5 (%)
-- | :--: | :--:  | :--: | :--:  | :--: | :--: 
`R50`            | 224x224 | 25.6 | 4.1 | 8.7 | 76.2 | 93.8
[`R50-max`](https://drive.google.com/file/d/1MoCdVLPau4XuI0BVGEwDkiwibpybqWTX/view?usp=sharing)    | 224x224 | 14.2 | 2.2 | 6.8 | 74.3 | 92.0  
[`R50-hybrid`](https://drive.google.com/file/d/1CyajEQUfWo9oetqcIhexVjfRXU-iGHk1/view?usp=sharing) | 224x224 | 17.3 | 2.6 | 7.3 | 77.1 | 93.1 
||||||
[`R50-max+`](https://drive.google.com/file/d/1TQfwMXENn7s78Myc58svKVpHi5OVsdGX/view?usp=sharing)    | 224x224 | 14.2 | 2.2 | 6.8 | 75.5 | 92.1  
[`R50-hybrid+`](https://drive.google.com/file/d/1f5fOwEItFITqzofvQ6d-kLLOTs-c1k3o/view?usp=sharing) | 224x224 | 17.3 | 2.6 | 7.3 | 78.0 | 93.5 


Network | Img size | Throughputs | Vanilla | +CutMix | +DeiT
-- | :--: | :--:  | :--: | :--:  | :--:
`R50`          | 224x224 | 962 / **112** | **76.2** | 77.6 | 78.8
[`ViT-S-max`](https://drive.google.com/file/d/19lfagLJDXWvVHcb8Qm_U4_A_Kr7qYLv7/view?usp=sharing)    |224x224 |  763 / 96 | 74.2 | 77.3 | 80.0
[`PiT-S-max`](https://drive.google.com/file/d/1S9JJM2WGtDtpo-6Me7Ak74nJOlOeBKeH/view?usp=sharing)     |224x224 |  **1000** / 92 |75.7 | **78.1** | **80.8**

#### Model load & evaluation
Example code of loading ``resnet50_hybrid`` without ``timm``:
```Python
import torch
from resnet_pf import resnet50_hybrid

model = resnet50_hybrid() 
model.load_state_dict(torch.load('./weight/checkpoint.pth'))
print(model(torch.randn(1, 3, 224, 224)))
```

Example code of loading ``pit_s_max`` with ``timm``:
   
```Python
import torch
import timm
import pit_pf
   
model = timm.create_model('pit_s_max', pretrained=False)
model.load_state_dict(torch.load('./weight/checkpoint.pth'))
print(model(torch.randn(1, 3, 224, 224)))
```    

Directly run each model can verify a single iteration of forward and backward of the mode.

#### Training 

Our **ResNet-based models** can be trained with any PyTorch training codes; we recommend [timm](https://github.com/rwightman/pytorch-image-models). We provide a sample script for training R50_hybrid with the standard 90-epochs training setup:


      python3 -m torch.distributed.launch --nproc_per_node=4 train.py ./ImageNet_dataset/ --model resnet50_hybrid --opt sgd --amp \
      --lr 0.2 --weight-decay 1e-4 --batch-size 256 --sched step --epochs 90 --decay-epochs 30 --warmup-epochs 3 --smoothing 0\


**Vision transformers (ViT and PiT) models** are also able to be trained with [timm](https://github.com/rwightman/pytorch-image-models), but we recommend the code [DeiT](https://github.com/facebookresearch/deit) to train with.  We provide a sample training script with the default training setup in the package:


      python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_s_max --batch-size 256 --data-path ./ImageNet_dataset/



## License

```
Copyright 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


### How to cite

```
@inproceedings{han2022learning,
    title={Learning Features with Parameter-Free Layers},
    author={Dongyoon Han and YoungJoon Yoo and Beomyoung Kim and Byeongho Heo},
    year={2022},
    journal={International Conference on Learning Representations (ICLR)},
}
```














