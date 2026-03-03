import cv2
import os
import numpy as np
import random
import string

# --- CONFIGURATION ---
INPUT_FOLDER = 'raw_images'
DATASET_FOLDER = 'dataset'
IMG_SIZE = 128  # We resize images to 128x128 so the AI can process them faster

# Create the output folders if they don't exist
os.makedirs(os.path.join(DATASET_FOLDER, 'clean'), exist_ok=True)
os.makedirs(os.path.join(DATASET_FOLDER, 'stego'), exist_ok=True)

def random_string(length):
    """Generates random garbage text to hide"""
    letters = string.ascii_letters + string.punctuation
    return ''.join(random.choice(letters) for _ in range(length))

def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def embed_message(image, secret_message):
    """Hides the message strictly in the Blue Channel LSB"""
    blue_channel = image[:, :, 0]
    bits = text_to_bits(secret_message)
    flat_blue = blue_channel.flatten()
    
    if len(bits) > len(flat_blue):
        return None # Message too big, skip
        
    for i in range(len(bits)):
        # Clear the LSB and insert the new bit
        flat_blue[i] = (flat_blue[i] & 254) | int(bits[i])
        
    reshaped_blue = flat_blue.reshape(blue_channel.shape)
    image[:, :, 0] = reshaped_blue
    return image

# --- THE FACTORY LOOP ---
print("Starting Data Generation...")
processed_count = 0

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        continue

    # 1. Standardize the size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 2. Save the "Clean" copy (Label 0)
    clean_name = f"clean_{processed_count}.png"
    cv2.imwrite(os.path.join(DATASET_FOLDER, 'clean', clean_name), img)
    
    # 3. Create and save the "Stego" copy (Label 1)
    # We fill about 50% of the image's capacity with secret data
    capacity = IMG_SIZE * IMG_SIZE // 8 
    msg_len = int(capacity * 0.5) 
    secret_msg = random_string(msg_len)
    
    stego_img = embed_message(img.copy(), secret_msg)
    
    if stego_img is not None:
        stego_name = f"stego_{processed_count}.png"
        cv2.imwrite(os.path.join(DATASET_FOLDER, 'stego', stego_name), stego_img)
        
        processed_count += 1
        print(f"Processed {filename} -> Created Clean/Stego pair #{processed_count}")

print(f"\nDone! Created {processed_count} pairs ({processed_count * 2} total images).")