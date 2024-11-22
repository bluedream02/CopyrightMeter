import os
import json
import numpy as np
from PIL import Image
import shutil
import torch
from transformers import CLIPTextModel, CLIPProcessor, CLIPModel


class CLIPSimilarityCalculator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    def calculate_image_similarity(self, image_path1, image_path2):
        inputs1 = self.load_and_preprocess_image(image_path1)
        inputs2 = self.load_and_preprocess_image(image_path2)

        with torch.no_grad():
            image_features1 = self.model.get_image_features(**inputs1).cpu().numpy()
            image_features2 = self.model.get_image_features(**inputs2).cpu().numpy()

        image_features1 /= np.linalg.norm(image_features1, axis=1, keepdims=True)
        image_features2 /= np.linalg.norm(image_features2, axis=1, keepdims=True)

        cosine_similarity = np.dot(image_features1, image_features2.T)
        return cosine_similarity.item()

    def calculate_text_image_similarity(self, image_path, text):
        inputs_image = self.load_and_preprocess_image(image_path)
        inputs_text = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs_image).cpu().numpy()
            text_features = self.model.get_text_features(**inputs_text).cpu().numpy()

        image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)
        text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)

        cosine_similarity = np.dot(image_features, text_features.T)
        return cosine_similarity.item()

    

def list_files(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


input_dir = "/path/input" # 原图


list1=list_files(input_dir)

CLIP_T_list=[]

model_path = "/path/clip-vit-base-patch32"
clip_calculator = CLIPSimilarityCalculator(model_path)

for i in list1:
    try:
        image_path1 = i
        text = "a photo of "+i.split('/')[-2].replace("_"," ")
        text_image_similarity = clip_calculator.calculate_text_image_similarity(image_path1, text)
        print("Cosine Similarity between the image and text:", text_image_similarity)
        CLIP_T_list.append(text_image_similarity)
    except Exception as e:
        print(f"Error calculating CLIP-I for items {i}, {j}: {e}")
        continue
    
               
CLIP_T_avg = np.mean(CLIP_T_list)
CLIP_T_std = np.std(CLIP_T_list)

data = {
    # "CLIP_T_list": CLIP_T_list,
    "CLIP_T_avg": CLIP_T_avg,
    "CLIP_T_std": CLIP_T_std,
}

print(data)