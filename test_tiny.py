import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from models.densenet import DenseNet
from PIL import Image, ImageDraw, ImageFont
from dataset import train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader, label_dict, label_word_dict, inverted_label_dict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def pil_to_cv2_font(pil_font):
    font_bytes = pil_font.tobytes()
    font_cv2 = np.frombuffer(font_bytes, dtype=np.uint8).reshape(pil_font.size[1], pil_font.size[0], 3)
    return font_cv2[:,:,0]

def main():
    print("Testing begin.\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseNet(num_classes=200)
    model = torch.load('DenseNet_Model_Tiny.pth')
    net.load_state_dict(model)
    net.to(device)
    net.eval()
    output_folder_path = 'D:/DeepLearning/CnnNet/tiny_imagenet_data/test/outputs'

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    with torch.no_grad():
        for images, image_paths in tqdm(test_dataloader):
            images = images.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)

            for img, path, pred in zip(images, image_paths, preds):
                img = img.cpu().numpy().transpose((1, 2, 0))
                img = np.clip((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0, 0, 255).astype(np.uint8)

                label = label_word_dict[inverted_label_dict[pred.item()]]

                img_with_label = Image.fromarray(img)
                draw = ImageDraw.Draw(img_with_label)
                
                font_path = "D:/DeepLearning/CnnNet/font.ttf"
                font_size = 24
                pil_font = ImageFont.truetype(font_path, size=font_size)

                text_position = (10, 30)

                draw.text(text_position, label, font=pil_font, fill=(0, 0, 0))

                img_with_label = np.array(img_with_label)

                filename = os.path.basename(path)
                output_path = os.path.join(output_folder_path, filename)
                cv2.imwrite(output_path, img_with_label)
    
    print("\nTesting complete.")

if __name__ == '__main__':
    main()
