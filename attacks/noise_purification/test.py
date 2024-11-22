import os 
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel


model_id = "/path/stable-diffusion-1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
image = Image.open("test/protected.jpg").convert("RGB")


# JPEG
from src.adversarial_purifier import JPEG
jpeg = JPEG().purify(pipe, image, target)

# Quantize
from src.adversarial_purifier import Quantize
quan = Quantize().purify(pipe, image, target)

# TVM
from src.adversarial_purifier import TVM
h = TVM().purify(None, image, None)


# IMPRESS
from src.adversarial_purifier import Impress
impress = Impress().purify(pipe, advdm, target)

# DiffPure
from src.adversarial_purifier import DiffusionPurifier
diffusion_purify = diffusion_pur(input_image_path="test/protected.jpg", model_path="models/256x256_diffusion_uncond.pt") 
image = diffusion_purify.generate()
