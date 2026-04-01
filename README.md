# Test-Time-Explainability
This repository contains the official implementation of the test-time explainability from the paper ["TTE-CAM: Built-in Class Activation Maps for Test-Time Explainability in Pretrained Black-Box CNNs"](https://arxiv.org/abs/2603.26885).


## Data
The code in this repository uses publicly available datasets for the [Diabetic Retinopathy Detection Challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) and the [RSNA Pneumonia dataset](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

## Training
The training was done following this repository [Soft-CAM: Making black box models self-explainable for medical image analysis](https://github.com/kdjoumessi/SoftCAM).

## Model weights
The models weights can be downloaded here
- [ResNet-50 trained for DR detection from the Kaggle dataset](https://drive.google.com/file/d/1vbTuuuTROdsFXRqMHkFFc1S51_gPWiIQ/view?usp=drive_link)
- [ResNet-50 trained for pneunomia detection from the RSNA dataset](xxxx)

## From black box to explainable model
[jupyter noteobok](TTE-CAM.ipynb)

