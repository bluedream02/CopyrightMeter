import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffusers import StableDiffusionPipeline
import torch


dataset="artist"
dataset="person"
folder = f'./output_images_concept_inv_model/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)

    pipeline = StableDiffusionPipeline.from_pretrained(f"./output_images_concept_inv_model/{dataset}/{subfolder}", torch_dtype=torch.float16,safety_checker=None).to("cuda")

    path= f"./output_images_concept_inv_image/{dataset}/{subfolder}"
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")
    for i in range(10):
        if dataset=="artist":
            image = pipeline(prompt="A painting in the style of "+subfolder.replace("-"," ")).images[0]
        else:
            image = pipeline(prompt="A photo of "+subfolder.replace("-"," ")).images[0]
        image.save(f"{path}/{i}.png")
