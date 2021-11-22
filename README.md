# DM6190 Assignment 2: Adversarial Unsupervised Domain Adaptation forCross-modality Cardiac Image Segmentation



## Team Member
* Zhao Ziyuan
* Ng Han Wei


## Abstract
Deep convolutional neural networks (DCNNs) achieve great success in 
medical image segmentation, but a model trained on the source domain always performs 
poorly on the target domain due to the severe domain shift. On the other hand, medical image annotations 
are costly and laborious which introduces the label scarcity problem on the source domain. Recently unsupervised 
domain adaptation (UDA) has become one popular topic in studies on cross-modality medical image segmentation, 
which aims to recover performance degradation when applying the well-trained model on one domain to unseen domains 
without annotations. In this work, we investigated the applications of adversarial learning on UDA, reviewed and 
implemented three representative adversarial unsupervised domain adaptation (AUDA) methods from different perspectives. 
Extensive experiments and analysis were carried out on MM-WHS 2017 dataset, demonstrating the effectiveness of
adversarial image and feature adaptation on cross-modality cardiac image segmentation.

## Reimplemented methods
* Image Adaptation - [CycleGAN](https://arxiv.org/abs/1703.10593)

* Feature Adaptation - [ADDA](https://arxiv.org/abs/1702.05464)

* Image + Feature Adaptation - [CyCADA](https://arxiv.org/abs/1711.03213)




## Setup

1. Follow official guidance to install [Pytorch](https://pytorch.org/).
2. Clone the repo
3. Install python requirements - pip install -r requirements.txt



## Data Preparation
MM-WHS: Multi-Modality Whole Heart Segmentation Challenge (MM-WHS 2018) dataset
http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/

The pre-processed data has been released from [PnP-AdaNet](https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation). The training data can be downloaded [here](https://drive.google.com/file/d/1m9NSHirHx30S8jvN0kB-vkd7LL0oWCq3/view). The testing CT data can be downloaded [here](https://drive.google.com/file/d/1SJM3RluT0wbR9ud_kZtZvCY0dR9tGq5V/view).
The testing MR data can be downloaded [here](https://drive.google.com/file/d/1Bm2uU4hQmn5L3GwXz6I0vuCN3YVMEc8S/view?usp=sharing).

    
    
    
## Image Adaptation

Image adaptation builds on the work on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

1) Load in transformed images in one folder
2) Run `bash train_fcn.sh`

## Feature Adaptation

1) Load in source images in one folder
2) Run 'bash train.sh' to create baseline model
3) Load pre-trained FCN/DRN network trained on source images
4) Load target images in separate folder
5) Run `bash train_fcn_adda.sh`


## Image + Feature Adaptation

1) Load pre-trained FCN/DRN network trained on transformed images (CycleGAN)
2) Load in transformed images in one folder
3) Load target images in separate folder
4) Run `bash train_fcn_adda.sh`


## Evaluation 
To evaluate the performance of the model,
Run eval_fcn_ct.py
1) Load in trained model
2) Specify npz folder containing test volumes


## Visualization
1) Load in trained model
2) Specify npz folder containing test volumes
3) Uncomment visualization portion of the code 





## Citation
If you find the codebase useful for your research, please cite the papers:
```
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}

@inproceedings{tzeng2017adversarial,
  title={Adversarial discriminative domain adaptation},
  author={Tzeng, Eric and Hoffman, Judy and Saenko, Kate and Darrell, Trevor},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7167--7176},
  year={2017}
}

@inproceedings{Hoffman_cycada2017,
       authors = {Judy Hoffman and Eric Tzeng and Taesung Park and Jun-Yan Zhu,
             and Phillip Isola and Kate Saenko and Alexei A. Efros and Trevor Darrell},
       title = {CyCADA: Cycle Consistent Adversarial Domain Adaptation},
       booktitle = {International Conference on Machine Learning (ICML)},
       year = 2018
}

@inproceedings{zhao2021mt,
  title={MT-UDA: Towards Unsupervised Cross-modality Medical Image Segmentation with Limited Source Labels},
  author={Zhao, Ziyuan and Xu, Kaixin and Li, Shumeng and Zeng, Zeng and Guan, Cuntai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={293--303},
  year={2021},
  organization={Springer}
}
```

## Acknowledgement

Part of the code is adapted from open-source codebase and original implementations of algorithms, 
we thank these author for their fantastic and efficient codebase:
* CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
* ADDA & CyCADA: https://github.com/jhoffman/cycada_release
