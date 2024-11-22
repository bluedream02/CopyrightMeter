import os 
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel


model_id = "/path/to/stable-diffusion-1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = Image.open("test/input.jpg").convert("RGB")
target = Image.open("test/target.jpg").convert("RGB")


# AdvDM
from src.adversarial_generator import AdvDM
advdm = AdvDM()
advdm_result = advdm.generate(pipe, image, target)

# Mist
from src.adversarial_generator import Mist
mist = Mist("/path/to/mist-main", "test/input.jpg", "output/misted_sample.jpg")
mist.generate()

# Glaze
from src.adversarial_generator import Glaze
glaze = Glaze()
glaze_result = glaze.generate(pipe, image, target)

# PhotoGuard
from src.adversarial_generator import EncoderAttackImg2Img
encoder_attack = EncoderAttackImg2Img()
encoder_attack_result = encoder_attack.generate(pipe, image)

from src.adversarial_generator import DiffusionAttack
encoder_attack_img2img = EncoderAttackImg2Img()

# Anti-DreamBooth
from src.adversarial_generator import AntiDreamBoothUse
anti_dreambooth_use = AntiDreamBoothUse("/path/to/Anti-DreamBooth-main", "attack_with_aspl")
anti=anti_dreambooth_use.generate()

