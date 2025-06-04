# Image classification: Wildlife recognition model
[![License](https://img.shields.io/github/license/Enishe1/WildVision)](https://github.com/Enishe1/WildVision/blob/main/LICENSE.txt)
[![GitHub stars](https://img.shields.io/github/stars/Enishe1/WildVision)](https://github.com/Enishe1/WildVision/stargazers)

WildVision is a wildlife species identification project developed as part of the Data Science and Artificial Intelligence course at the Faculty of Electrical Engineering, University of Sarajevo. The project employs image classification models to identify various wildlife species, utilizing a dataset of wildlife images. The system aims to assist in the automated classification of wildlife, contributing to conservation efforts and biodiversity monitoring.<br>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

**Team Members:**
* [Enis Herac] - ([GitHub Profile](https://github.com/Enishe1))
* [Dženis Muhović] - ([GitHub Profile](https://github.com/dmuhovic1))
<br><br>

**What to expect from WildVision:**
* [WildVision's](https://github.com/Enishe1/WildVision/blob/main/src/train.py) first model choice was ResNet-18, the shallowest of the ResNet family. With this model, we achieved around 50% accuracy,which was insufficient for our image classification task. Therefore, we switched to ResNet-50,
   which has significantly more layers and higher accuracy. ResNet-50 uses bottleneck blocks that make it deeper and more efficient than ResNet-18. It also has greater capacity to learn complex features,
   making it superior in almost every aspect. Using ResNet-50, we improved our validation accuracy to about 70%, a solid performance for image classification models.  

  But still, we are not satisfied with this performance. In the future we want to improve the validation accuracy to atleast 80%.

## Installation

Clone the repository:
```bash
git clone https://github.com/Enishe1/WildVision?tab=readme-ov-file
```

## Datasets

[Image Recognition Model](https://www.kaggle.com/datasets/pytorch/resnet50/data)<br>
The dataset used for training and evaluation of detection model is available on pypi.org. 

## Training 
Training was done on ResNet-50 model. The training process was optimised with recommended optimizer for that specific dataset.
Following are the training parameters and results of all models. 

## Training Progress (ResNet-50)

| **Epoch** | **Train Accuracy** | **Validation Accuracy** | **Train Loss** | **Validation Loss** |
|:---------:|:------------------:|:-----------------------:|:--------------:|:-------------------:|
|     1     |       ~0.40        |         ~0.38           |     ~2.40      |       ~2.45         |
|     8     |       ~0.55        |         ~0.52           |     ~1.60      |       ~1.70         |
|    16     |       ~0.65        |         ~0.62           |     ~1.20      |       ~1.30         |
|    25     |       ~0.72        |         ~0.70           |     ~0.95      |       ~1.05         |

---

##  Model Summary

| **Model**  | **Epochs** | **Batch Size** | **Learning Rate** | **Optimizer** | **Dropout** | **Validation Accuracy** |
|:----------:|:----------:|:--------------:|:-----------------:|:-------------:|:-----------:|:------------------------:|
| ResNet-50  |     25     |      32        |      0.001        |     Adam      |     0.2     |           70%           |



## Results
Below are some key performance indicators and visualizations from our training runs.


### 1. Detection with Large YOLOv11 model

* **Per-class accuracy:**

    <img src="https://raw.githubusercontent.com/Enishe1/WildVision/main/plots/per_class_accuracy.png" alt="Per Class Accuracy" width="720"/>


* **Training history:**

    <img src="https://raw.githubusercontent.com/Enishe1/WildVision/main/plots/training_history.png" alt="Training History" width="720"/>

 * **Confusion Matrix:**

  <img src="https://raw.githubusercontent.com/Enishe1/WildVision/main/plots/confusion_matrix.png" alt="Confusion Matrix" width="720"/>


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the [![License](https://img.shields.io/github/license/Enishe1/WildVision)](https://github.com/Enishe1/WildVision/blob/main/LICENSE.txt) file for details.
