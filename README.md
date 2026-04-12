# Test-Time-Explainability
This repository contains the official implementation of the test-time explainability from the paper ["TTE-CAM: Built-in Class Activation Maps for Test-Time Explainability in Pretrained Black-Box CNNs"](https://arxiv.org/abs/2603.26885).


## Data
The code in this repository uses publicly available datasets from the [Diabetic Retinopathy Detection Challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) and the [RSNA Pneumonia dataset](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

## Training
The training was done following the SoftCAM repository for [Making black box models self-explainable for medical image analysis](https://github.com/kdjoumessi/SoftCAM).

## Model weights
The models weights can be downloaded here
- [ResNet-50 trained for DR detection from the Kaggle dataset](https://drive.google.com/file/d/1vbTuuuTROdsFXRqMHkFFc1S51_gPWiIQ/view?usp=drive_link)
- [ResNet-50 trained for pneunomia detection from the RSNA dataset](https://drive.google.com/file/d/1rBTQxXO9pAMRUKknFKWkA-SZvgAPlUHj/view?usp=drive_link)

## From black box to test time explainable model
The procedure for generating built-in CAM-based explanations from standard black-box CNNs is described here: [From black-box to TTE-CAM](TTE-CAM.ipynb)

## Acknowledge
This implementation was inspired by: 
- [Class-Activation-Maps with PyTorch and a ResNet model](https://github.com/maubreville/ClassActivationMaps_PyTorch/blob/master/ClassActivationMaps_Demo_Resnet18.ipynb)
- [SoftCAM: Making black box models self-explainable for medical image analysis](https://github.com/kdjoumessi/SoftCAM)

## Reference
```
@article{djoumessi2026tte,
  title={TTE-CAM: Built-in Class Activation Maps for Test-Time Explainability in Pretrained Black-Box CNNs},
  author={Djoumessi, Kerol and Berens, Philipp},
  journal={arXiv preprint arXiv:2603.26885},
  year={2026}
}

```