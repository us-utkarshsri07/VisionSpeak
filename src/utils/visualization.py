import matplotlib.pyplot as plt
import cv2
import numpy as np


def visualize_attention(image_path, caption_words, attentions):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    for i in range(len(caption_words)):
        plt.figure(figsize=(6, 6))

        attn = attentions[i].reshape(7, 7)
        attn = cv2.resize(attn, (w, h))
        attn = attn / (attn.max() + 1e-8)

        plt.imshow(image)
        plt.imshow(attn, cmap="jet", alpha=0.5)
        plt.title(caption_words[i])
        plt.axis("off")
        plt.show()
