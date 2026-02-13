# StegoVision: Deep Learning-Based Steganalysis Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📌 Project Overview
StegoVision is an AI-powered security tool designed to detect **LSB (Least Significant Bit) Steganography** in digital media. 

While traditional steganography hides data by manipulating the noise floor of an image, standard statistical attacks often fail to detect sophisticated embedding. StegoVision leverages **Convolutional Neural Networks (CNNs)** combined with **Information Theoretic** feature extraction (Entropy & Bit-Plane Analysis) to identify statistical anomalies that indicate the presence of hidden payloads.

## 🚀 Key Features
* **Bit-Plane Slicing:** Automated decomposition of images into 8-bit planes to isolate high-frequency noise.
* **Entropy Analysis:** Calculates local Shannon Entropy to detect non-random distribution in LSB layers.
* **Residual Learning:** Utilizes a custom CNN architecture to learn the "residual noise" signature of steganographic embedding.
* **Automated Dataset Generation:** Includes a pipeline to generate synthetic "Stego" datasets for supervised learning.

## 🛠️ Technical Architecture
The system operates on a 3-stage pipeline:
1.  **Preprocessing:** - Images are converted to raw bit streams.
    - High-pass filtering is applied to suppress image content and highlight noise.
2.  **Feature Engineering:**
    - **Spatial Domain:** LSB extraction.
    - **Frequency Domain:** DCT (Discrete Cosine Transform) coefficients analysis.
3.  **Classification:**
    - A custom 5-layer CNN processes the noise map to classify images as `Clean` or `Stego`.

## 🔧 Installation & Usage
```bash
# Clone the repository
git clone [https://github.com/yourusername/StegoVision.git](https://github.com/yourusername/StegoVision.git)

# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data
python src/generator.py --count 1000