# 🖼️ VisionSpeak  
### Attention-Based Image Captioning System (CNN + LSTM + Additive Attention)

<p align="center">
  <img src="glimpse.png" width="100%">
</p>

---

## 📌 1. Project Overview

**VisionSpeak** is a deep learning–based image captioning system that generates natural language descriptions for images.

The system learns to:

- Extract spatial visual features from an image  
- Focus on different regions of the image while generating each word  
- Produce coherent captions using sequence modeling  
- Provide interpretable attention heatmaps  

### 🎯 Core Objective

> Given an image → Generate a grammatically meaningful caption describing the scene.

This project implements a classical **Encoder–Decoder architecture with Attention**.

---

## 🧠 2. Key Concepts (Important Terms)

### 🔹 Convolutional Neural Network (CNN)

A **CNN** is a deep learning architecture designed for image understanding.

It extracts hierarchical spatial features using convolutional filters.

In this project:

- A pretrained CNN is used to extract image features.
- Each image is converted into a spatial feature map of shape:






