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
- Each image is converted into a spatial feature map of shape: ```(49, 2048)```

---

This represents:

- 7 × 7 spatial grid  
- 2048 feature dimensions per region  

These 49 feature vectors allow spatial attention.

---

### 🔹 Long Short-Term Memory (LSTM)

An **LSTM** is a type of Recurrent Neural Network (RNN) used for modeling sequential data.

It maintains:

- Hidden state (short-term memory)  
- Cell state (long-term memory)  

In this project:

- The LSTM generates captions word-by-word.
- At each timestep, it predicts the next word conditioned on:
  - Previous words
  - Visual context

---

### 🔹 Attention Mechanism

Attention allows the model to focus on different parts of the image while generating each word.

Instead of compressing the image into a single vector, attention:

- Computes weights over all 49 spatial regions  
- Produces a weighted combination (context vector)  
- Uses this context to generate the next word  

This makes the model:

- More accurate  
- More interpretable  

---

## 📐 3. Problem Definition

Image captioning is a multimodal task combining:

- Computer Vision (understanding image content)  
- Natural Language Processing (generating text)  

Given image \( I \), generate caption:

\[
S = (w_1, w_2, ..., w_T)
\]

The probability model:

\[
P(S|I) = \prod_{t=1}^{T} P(w_t | w_{1:t-1}, I)
\]

The system learns to predict the next word conditioned on:

- Previous words  
- Visual features  

---

## 🏗️ 4. Architecture

### 🔹 4.1 Encoder

Instead of training a CNN from scratch, **pre-extracted CNN features** are used.

Each image representation:```Shape: (49, 2048)```








