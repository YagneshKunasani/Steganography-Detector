import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- THE UPGRADED AI BRAIN (Must match train.py exactly) ---
class StegoNet(nn.Module):
    def __init__(self):
        super(StegoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # The Forensic Layer: Strips the picture, leaves the LSB noise!
        x = torch.round(x * 255.0) % 2.0
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- UI SETUP ---
st.set_page_config(page_title="StegoVision", page_icon="👁️", layout="centered")
st.title("👁️ StegoVision: Deep Learning Steganalysis")
st.markdown("Upload a 128x128 image to detect hidden steganographic data in the LSB layers.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image from your 'dataset' folder...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Notice: NO resizing here! Resizing destroys the LSB data.
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    try:
        import os
        # 1. Get the exact folder this app.py script is living in
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # 2. Build the exact path to the smart model
        model_path = os.path.join(BASE_DIR, 'stego_model.pth')
        
        # Display where we are loading from just to be sure
        st.write(f"📂 *Loading AI from: {model_path}*")
        
        model = StegoNet()
        # 3. Load the model using the strict absolute path
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence = probabilities[prediction].item() * 100
            
        if prediction == 1:
            st.error(f"🚨 **ALERT: Hidden Data Detected!** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ **CLEAN: No Hidden Data Found.** (Confidence: {confidence:.2f}%)")
            
    except RuntimeError as e:
        st.error(f"Image Size Error: Expected 128x128. Got {image.size}.")
    except FileNotFoundError:
        st.warning(f"Model weights not found at {model_path}. Please train the model first.")