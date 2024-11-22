import os
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "/path/model/stable-diffusion-1-5",
    safety_checker = None,
    torch_dtype=torch.float16)
pipe = pipe.to("cuda")


dataset="artist"
folder = f'/path/input/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    pipe.safety_concept=subfolder.replace("-"," ")
    path=f"./np_erased_image/{dataset}/{subfolder}"
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")
    for i in range(10):
        t = subfolder.replace("-"," ").replace("_"," ")
        prompt = "A painting in the style of "+ t
        negative_prompt = t
        image = pipe(prompt=prompt, negative_prompt=t).images[0]
        image.save(f"{path}/{i}.png")
