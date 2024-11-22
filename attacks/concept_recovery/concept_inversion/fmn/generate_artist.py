import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffusers import StableDiffusionPipeline
import torch


dataset="artist"

folder = f'./fmn_model/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)

    pipeline = StableDiffusionPipeline.from_pretrained("./output", torch_dtype=torch.float16,safety_checker=None).to("cuda")

    path= f"./fmn_img/{dataset}/{subfolder}"
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")
    for i in range(10):
        image = pipeline(prompt="A painting in the style of "+subfolder.replace("-"," ")).images[0]
        image.save(f"{path}/{i}.png")