# Lane Detection under Low-illumination Condition by Enhanced Feature Learning

## Introduction

This project focuses on improving lane detection performance in low-illumination conditions, such as nighttime, by integrating an image enhancement module with lane detection algorithms. The framework consists of two main modules: an image enhancement module to process low-visibility images and a detection module to detect lane parameters.

## Abstract

In low-illumination conditions, images often suffer from low contrast, low brightness, and noise, making lane detection challenging. This project presents a framework that enhances image features before feeding them into a lane detection network. The proposed method shows significant improvements in performance, achieving a 1.32% and 0.67% increase in F1-measurement for two state-of-the-art detectors, RESA and LSTR, respectively.

## Features

- **Low-Light Image Enhancement:** Utilizes Zero-Reference Deep Curve Estimation (ZeroDCE) for image preprocessing.
- **Lane Detection Models:** Supports RESA and LSTR detectors.
- **Loss Collaboration:** Combines enhancement and detection loss functions for end-to-end training.

## Methodology

### Framework Overview

The framework includes:
1. **Enhancement Module:** Adjusts low-light images using an unsupervised, lightweight convolutional neural network.
2. **Detection Module:** Detects lanes using either RESA or LSTR models.

### Low-Light Image Enhancement

Enhances images using a curve estimation problem solved by a lightweight CNN, iteratively adjusting pixel values to improve visibility.

### Lane Detection Model

- **RESA:** Uses a Recurrent Feature-Shift Aggregator for pixel-wise lane detection.
- **LSTR:** Predicts lane shape parameters directly using a transformer-based network.

## Experiments and Results

### Dataset and Evaluation Metrics

Evaluated on the CULane dataset, which includes various illumination scenarios. The main metric is F1-measurement, considering Precision and Recall.

### Training Setup

- **Enhancement Module:** Pre-trained on multi-illumination images.
- **Lane Detectors:** Trained with the enhanced images, using separate setups for RESA and LSTR.

### Performance

- **Night Scenario:** Both detectors show improved performance with the enhancement module.
- **Different Illumination Conditions:** Robust performance across normal, shadow, and dazzle scenarios.

### Ablation Study

The best performance is achieved when enhancement and lane detection losses are balanced equally.

## Conclusion

The proposed framework significantly improves lane detection in low-illumination conditions. Future work will focus on enhancing the performance of the image enhancement module and applying the method to other vision tasks.