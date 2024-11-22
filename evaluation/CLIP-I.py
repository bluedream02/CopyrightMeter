import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import csv

class CLIPSimilarityCalculator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
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



folder1 = "/path/input"
folder2 = "/path/output"


CLIP = CLIPSimilarityCalculator("/path/clip-vit-base-patch32")
CLIPI = []

for f1 in os.listdir(folder1):
    # print(f1)
    for pics_f1 in os.listdir(folder1+"/"+f1):
        pic1 = folder1+"/"+f1+"/"+pics_f1
        for pics_f2 in os.listdir(folder2+"/"+f1):
            pic2 = folder2+"/"+f1+"/"+pics_f2
            # print(pic1)
            # print(pic2)
            if pics_f1 == pics_f2:
                temp = CLIP.calculate_image_similarity(pic1, pic2)
                CLIPI.append(temp)

avg_clip_i = np.mean(CLIPI)
var_clip_i = np.var(CLIPI)

data = {
    "CLIP_I_avg": avg_clip_i,
    "CLIP_I_std": var_clip_i,
}

print(data)