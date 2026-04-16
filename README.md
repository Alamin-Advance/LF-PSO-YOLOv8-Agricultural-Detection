# LF-PSO-YOLOv8 for Agricultural Object Detection

This repository contains the implementation of my seminar research: **Integrating Levy Flight Particle Swarm Optimization (LF-PSO) with YOLOv8 for Enhanced Agricultural Object Detection**.

The project applies metaheuristic optimization to automatically tune YOLOv8 hyperparameters for improved wheat head detection in real-world field conditions.

---

## 📊 Key Results

| Metric | Baseline YOLOv8 | LF-PSO-YOLOv8 | Improvement |
|--------|----------------|---------------|-------------|
| mAP₅₀₋₉₅ | 0.5188 | **0.5313** | **+6.4%** |
| Precision | 0.8998 | 0.8984 | — |
| Recall | 0.8682 | **0.8805** | **+1.4%** |
| mAP₅₀ | 0.9281 | **0.9325** | **+0.5%** |

- **Dataset:** Global Wheat Head Detection (GWHD)
- **Image Size:** 896×896 pixels
- **Optimized Hyperparameters:** learning rate, momentum, IoU threshold, augmentation factors

---

## 🚀 How to Run

### Option A: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l-Jc6bDS6Njxw_r2PWQwIsa1TSTVIFyG?hl=en#scrollTo=zB_TTJbsH-bt)

Click the badge above to open the notebook. The code will automatically mount your Google Drive and access the required files.

### Option B: Local Machine

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/LF-PSO-YOLOv8-Agricultural-Detection.git
cd LF-PSO-YOLOv8-Agricultural-Detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook LF_PSO_YOLOv8_Wheat_Detection.ipynb

##Repository Structure

LF-PSO-YOLOv8-Agricultural-Detection/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LF_PSO_YOLOv8_Wheat_Detection.ipynb # Main Colab notebook
├── gwhd.yaml                          # Dataset configuration
│
└── results/                           # Sample output (if included)
    ├── convergence_curve.png
    └── sample_detections.png

##📝 Dataset
The Global Wheat Head Detection (GWHD) dataset is used for this project.

Source: Kaggle Global Wheat Detection
Images: 3,000+ high-resolution RGB images
Annotations: ~120,000 wheat head bounding boxes
Geographic Coverage: France, Japan, Canada, Australia, UK

#Setup Instructions
Download the dataset from Kaggle

Organize files in the following structure:
SeminarYOLO/
├── gwhd.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/


##🔬 Methodology
Optimization Framework
The project combines:
-YOLOv8 – Anchor-free object detector with C2f backbone and decoupled head
-Particle Swarm Optimization (PSO) – Swarm intelligence for hyperparameter search
-Levy Flight (LF) – Heavy-tailed random walks to escape local optima

Optimized Hyperparameters
Parameter	Lower Bound	Upper Bound	Description
lr0	0.001	0.020	Initial learning rate
lrf	0.05	0.30	Final learning rate factor
momentum	0.90	0.98	SGD momentum
weight_decay	0.0001	0.0020	L2 regularization
iou	0.45	0.70	IoU threshold
hsv_h	0.00	0.05	Hue augmentation
hsv_s	0.50	0.90	Saturation augmentation

##📈 Results Visualization
Convergence Comparison
The LF-PSO curve shows smoother, monotonic reduction of fitness compared to standard PSO, confirming improved stability.

-Sample convergence plot available in the results/ folder.
-Sample Detections
-LF-PSO-YOLOv8 produces more accurate bounding boxes, especially in:
-Dense canopy areas
-Shadowed regions

#Overlapping wheat heads
hsv_v	0.40	0.90	Value augmentation
mosaic	0.0	1.0	Mosaic augmentation probability
LF-PSO Parameters
Parameter	Value	Description
Population size	6	Number of particles
Iterations	10	Optimization cycles
Inertia (ω)	0.6	Velocity memory
c₁, c₂	1.5	Cognitive and social factors
β (Levy)	1.5	Stability index
Step size	0.02	Perturbation magnitude

#📚 Related Publications
Integrating Levy Flight Particle Swarm Optimization with YOLOv8 for Enhanced Agricultural Object Detection (Seminar paper, not published yet)

##👨‍💻 Author
Md Al Amin Hossain
PhD Candidate, Information Technology Engineering
Selçuk University, Konya, Turkiye
📧 alaminh1411@gmail.com

##📄 License
This project is for academic research purposes. Please contact the author for permission to use or reproduce.

#🙏 Acknowledgments
-Supervisors: Dr. Tahir Sag, Dept. of Computer Engineering, Selçuk University, Konya, Turkiye
-Dataset: Global Wheat Head Detection (GWHD) consortium
-Ultralytics team for YOLOv8 implementation

##📧 Contact
For questions or collaboration inquiries, please email me at alaminh1411@gmail.com
