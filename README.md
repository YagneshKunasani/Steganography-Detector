# StegoVision: Deep Learning-Based Steganalysis Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📌 Project Overview
StegoVision is an end-to-end AI security tool and web application designed to detect **LSB (Least Significant Bit) Steganography** in digital media. 

While traditional steganography hides data by manipulating the noise floor of an image, standard Convolutional Neural Networks (CNNs) often fail to detect it because the visual "picture" drowns out the microscopic LSB noise. StegoVision solves this by utilizing a custom PyTorch **Forensic Extraction Layer** to strip away image content, allowing the CNN to strictly analyze the structural anomaly of the 0th Bit-Plane.

## 🚀 Key Features
* **Interactive Web UI:** Deployed a user-friendly frontend using Streamlit for real-time steganography detection and confidence scoring.
* **Automated Data Pipeline:** Includes a custom generator script to bulk-inject secret payloads into clean images, creating synthetic supervised learning datasets on the fly.
* **Forensic Preprocessing:** Mathematically isolates the Least Significant Bit using tensor manipulation: $x = \text{round}(x \times 255.0) \pmod{2.0}$.
* **Deep Learning Classifier:** A lightweight, optimized CNN architecture built in PyTorch to classify residual noise signatures as `Clean` or `Stego`.

## 🛠️ Technical Architecture
The repository is structured as a complete Software Engineering pipeline:
1. **`injector.py`**: The raw LSB manipulation script that converts strings to binary and embeds them into the blue channel of target images.
2. **`data_generator.py`**: The factory script that processes raw images into uniform 128x128 datasets, automatically generating matched Clean/Stego pairs.
3. **`train.py`**: The PyTorch training loop that feeds the generated datasets through the CNN, utilizing a custom learning rate and batch size to prevent model collapse, saving the final weights to `.pth`.
4. **`app.py`**: The Streamlit application that provides the inference interface.

## 🔧 Installation & Usage

**1. Clone the repository and install dependencies**
```bash
git clone [https://github.com/YagneshKunasani/Steganography-Detector.git](https://github.com/YagneshKunasani/Steganography-Detector.git)
cd Steganography-Detector
pip install -r requirements.txt

**2. Generating the Dataset**
Run python data_generator.py in the terminal

**3. Train the model**
Run python train.py in the terminal to the generate the stego_model.pth file

**4. Test AI in your browser**
Run streamlit run app.py in the terminal to test the model in your browser.

