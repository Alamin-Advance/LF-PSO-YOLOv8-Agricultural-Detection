# LF-PSO-YOLOv8 for Agricultural Object Detection

This repository contains the implementation of my seminar research on integrating **Levy Flight Particle Swarm Optimization (LF-PSO)** with **YOLOv8** for enhanced agricultural object detection.

## Key Results
- **Dataset:** Global Wheat Head Detection (GWHD)
- **Best mAP₅₀₋₉₅:** 0.5313 (6.4% improvement over baseline YOLOv8)
- **Optimized hyperparameters:** learning rate, momentum, IoU threshold, augmentation factors

## Code
The full implementation is available in the Colab notebook: https://colab.research.google.com/drive/1l-Jc6bDS6Njxw_r2PWQwIsa1TSTVIFyG?hl=en#scrollTo=FtbDCjNLB4ZZ

## Files
- `LF_PSO_YOLOv8_Wheat_Detection.ipynb` - Main training and optimization notebook

## Requirements
- Python 3.12+
- ultralytics
- numpy
- matplotlib
- pandas
- pyswarms (for PSO implementation)

## Author
Md Al Amin Hossain - PhD Student at Selcuk University, Turkiye
Email: alaminh1411@gmail.com
