# @title Load the SLD Pipeline and display the default safety concept
import os
from src.sld import SLDPipeline
import torch
    


#stable_diffusion_src_path = "CompVis/stable-diffusion-v1-4"
stable_diffusion_src_path = "/path/model/stable-diffusion-1-5"
device='cuda'

pipe = SLDPipeline.from_pretrained(
    stable_diffusion_src_path,
    safety_checker = None,
).to(device)
gen = torch.Generator(device=device)




dataset="artist"
folder = f'/path/model/dataset/update_lmx/origin_data/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    pipe.safety_concept=subfolder.replace("-"," ")
    path=f"./sld_erase_img/{dataset}/{subfolder}"
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")
    for i in range(10):
        image = pipe(prompt="A painting in the style of "+subfolder.replace("-"," ")).images[0]
        image.save(f"{path}/{i}.png")


    
        
