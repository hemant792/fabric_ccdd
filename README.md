# Fabric Construction, Count, and Defect Detection (CCD)

## Overview
This repository contains a Python-based solution for analyzing fabric quality, focusing on **construction estimation** (thread detection and alignment) using image processing and machine learning. Developed as part of my work at Zyod Commerce Private Limited and my M.Tech in Robotics at IIT Delhi, this project leverages computer vision and deep learning to automate fabric analysis. The current implementation handles construction (warp/weft thread), with plans to extend to count estimation and defect detection.

## Features
- **Construction Analysis**: Detects warp and weft thread counts and orientations using advanced image processing techniques.
- **Machine Learning Integration**: Utilizes a pre-trained model (`thread_verification.pth`) for thread quality prediction.
- **Parallel Processing**: Optimizes performance with multiprocessing for large datasets.
- **Scalable Design**: Modular code for future enhancements (e.g., count estimation, defect detection).

## Prerequisites
- **Python 3.8+**
- **Dependencies**:
  - `numpy`
  - `pandas`
  - `opencv-python`
  - `torch`
  - `numba`
  - `matplotlib`
  - `gdown` (for model download)
- **Hardware**: GPU recommended for model inference (CUDA support).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hemant792/fabric_ccdd.git
   cd fabric_ccdd
