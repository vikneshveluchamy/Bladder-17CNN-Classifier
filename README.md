Endoscopic Bladder Tissue Classification — 17-Layer CNN

This repository presents a deep-learning project for automatic classification of endoscopic bladder tissue images into four classes:

HGC – High-Grade Carcinoma

LGC – Low-Grade Carcinoma

NST – Non-Specific Tissue

NTL – Normal Tissue Layer

The model is a 17-Layer Convolutional Neural Network (CNN) inspired by medical deep-learning research and trained on real endoscopic tissue images.

A fully deployed interactive demo is available on Hugging Face:

👉 Live Demo: https://huggingface.co/spaces/vikneshveluchamy/Bladder-17CNN-Classifier

1. Project Overview

The aim of this work is to assist clinicians by automatically identifying bladder tissue types during endoscopic procedures.
The model is lightweight, fast, and suitable for deployment in medical-AI prototypes.

2. Model Architecture (17-Layer CNN)
Input (224×224×3)
↓
Conv2D → ReLU  
Conv2D → ReLU  
MaxPool  
↓
Conv2D → ReLU  
Conv2D → ReLU  
MaxPool  
↓
Conv2D → ReLU  
Conv2D → ReLU  
Conv2D → ReLU  
MaxPool  
↓
Conv2D → ReLU  
Conv2D → ReLU  
MaxPool  
↓
Conv2D → ReLU  
Conv2D → ReLU  
Conv2D → ReLU  
MaxPool  
↓
Flatten  
Linear → ReLU → Dropout  
Linear → Output (4 classes)

3. Dataset Structure

The dataset used in training follows:

bladder_dataset/
 ├── HGC/
 ├── LGC/
 ├── NST/
 └── NTL/


Training, validation, and test sets were created using stratified splitting.

4. Training Summary

Optimizer: Adam

Learning rate: 0.0001

Epochs: 50+

Loss: CrossEntropyLoss

Input size: 224×224

Achieved accuracy: ≈90% (paper-level range depending on preprocessing)

Training was performed on Google Colab (CPU/GPU fallback).

5. How to Run the Model Locally
Install dependencies
pip install -r requirements.txt

Run the Gradio App
python app.py


A browser window will open allowing you to upload images and get predictions.

6. File Structure
Bladder-17CNN-Classifier/
│
├── app.py                     # Gradio interface for inference
├── model.py                   # 17-Layer CNN model architecture
├── classes.json               # Class label mapping
├── final_bladder17.pth        # Trained model weights (not included if >25MB)
├── endoscopic_blader_tissues.ipynb  # Full training notebook
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation

7. Use Cases

Research in bladder cancer detection

Medical image classification

Clinical decision-support prototypes

Deep-learning education and demonstration

8. Author

Viknesh Veluchamy
B.Tech — Artificial Intelligence & Data Science
Research Interest: Medical Imaging • CNN • Deep Learning • AI for Healthcare

LinkedIn: https:https://www.linkedin.com/in/viknesh-v/
HuggingFace: https://huggingface.co/vikneshveluchamy

9. License

This project is for academic and research use only.
Not intended for clinical diagnosis.
