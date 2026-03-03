import cv2
import numpy as np

def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return bits.zfill(8*((len(bits) + 7) // 8))

def hide_message(image_path, secret_text, output_path):
    image = cv2.imread(image_path)

    full_message = secret_text + "#####"

    message_bits = text_to_bits(full_message)
    print(f"Hiding {len(message_bits)} bits of data")

    max_capacity = image.shape[0]*image.shape[1]*3
    if len(message_bits) > max_capacity:
        print("Error: Message is too long.")
        return
    
    flat_img = image.flatten()

    for i in range(len(message_bits)):
        current_val = flat_img[i]

        bit_to_hide = int(message_bits[i])

        flat_img[i] = (current_val & 254) | bit_to_hide

    stego_image = flat_img.reshape(image.shape)

    cv2.imwrite(output_path, stego_image)
    print(f"Successfully secret message is saved as {output_path}")

base_message = "My novel is inspired from The Novel's Extra"
massive_message = base_message * 800

hide_message('test_image.jpg',massive_message, 'stego_image.png')