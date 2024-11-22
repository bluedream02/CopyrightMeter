import os

import argparse
import yaml

import logging
import shutil
import numpy as np
from PIL import Image 
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

import torch
import torch.optim as optim
import torchvision.transforms as transforms
device = torch.device('cuda:1')
from diffusers import DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor

from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import *
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim

import json
import lpips
logging.info(f'===== Load Config =====')

with open('./example/config/config.yaml', 'r') as file:
    cfgs = yaml.safe_load(file)

formatted_json = json.dumps(cfgs, indent=4)
#logging.info(formatted_json)

logging.info(f'===== Init Pipeline =====')
if cfgs['w_type'] == 'single':
    wm_pipe = GTWatermark(device, w_channel=cfgs['w_channel'], w_radius=cfgs['w_radius'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))
elif cfgs['w_type'] == 'multi':
    wm_pipe = GTWatermarkMulti(device, w_settings=cfgs['w_settings'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))

scheduler = DDIMScheduler.from_pretrained(cfgs['model_id'], subfolder="scheduler")
pipe = WMDetectStableDiffusionPipeline.from_pretrained(cfgs['model_id'], scheduler=scheduler).to(device)
pipe.set_progress_bar_config(disable=True)

lpips_model = lpips.LPIPS(net="alex").to(device)



dataset="artist"
#dataset="person"
#dataset="CustomConcept"
folder = f'/path/input/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    subfolder_path = os.path.join(folder, subfolder)
    output_dict = {
        'Detect Prob': [],
        'SSIM': [],
        'PSNR': [],
        'LPIPS': [],
        'L2 distance': []
    }

    if os.path.isdir(subfolder_path):
        for image_file in os.listdir(subfolder_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')): 
                image_path = os.path.join(subfolder_path, image_file)
                print("Image path:", image_path)

                gt_img_tensor = get_img_tensor(image_path, device)

                wm_path=f"/path/output/{dataset}/{subfolder}"
                
                
                # Step 1: Get init noise
                def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
                    # DDIM inversion from the given image
                    img_latents = pipe.get_image_latents(img_tensor, sample=False)
                    reversed_latents = pipe.forward_diffusion(
                        latents=img_latents,
                        text_embeddings=text_embeddings,
                        guidance_scale=guidance_scale,
                        num_inference_steps=50,
                    )
                    return reversed_latents

                empty_text_embeddings = pipe.get_text_embedding('')
                init_latents_approx = get_init_latent(gt_img_tensor, pipe, empty_text_embeddings)

                # Step 2: prepare training
                init_latents = init_latents_approx.detach().clone()
                init_latents.requires_grad = True
                optimizer = optim.Adam([init_latents], lr=0.01)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.3) 

                totalLoss = LossProvider(cfgs['loss_weights'], device)
                loss_lst = [] 

                # Step 3: train the init latents
                for i in range(cfgs['iters']):
                    #logging.info(f'iter {i}:')
                    init_latents_wm = wm_pipe.inject_watermark(init_latents)
                    if cfgs['empty_prompt']:
                        pred_img_tensor = pipe('', guidance_scale=1.0, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
                    else:
                        pred_img_tensor = pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
                    loss = totalLoss(pred_img_tensor, gt_img_tensor, init_latents_wm, wm_pipe)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    loss_lst.append(loss.item())
                    # save watermarked image
                    if (i+1) in cfgs['save_iters']:
                        os.makedirs(wm_path, exist_ok=True)
                        path = os.path.join(wm_path, f"{image_file.split('.')[0]}_{i+1}.png")
                        save_img(path, pred_img_tensor, pipe)
                torch.cuda.empty_cache()



                # hyperparameter
                ssim_threshold = cfgs['ssim_threshold']
                wm_img_path = os.path.join(wm_path, f"{image_file.split('.')[0]}_{cfgs['save_iters'][-1]}.png")
                wm_img_tensor = get_img_tensor(wm_img_path, device)
                ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
                logging.info(f'Original SSIM {ssim_value}')

                def binary_search_theta(threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
                    for i in range(max_iter):
                        mid_theta = (lower + upper) / 2
                        img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
                        ssim_value = ssim(img_tensor, gt_img_tensor).item()

                        if ssim_value <= threshold:
                            lower = mid_theta
                        else:
                            upper = mid_theta
                        if upper - lower < precision:
                            break
                    return lower

                optimal_theta = binary_search_theta(ssim_threshold, precision=0.01)
                logging.info(f'Optimal Theta {optimal_theta}')

                img_tensor = (gt_img_tensor-wm_img_tensor)*optimal_theta+wm_img_tensor

                ssim_value = ssim(img_tensor, gt_img_tensor).item()
                psnr_value = compute_psnr(img_tensor, gt_img_tensor)


                
                lpips_distance = lpips_model(img_tensor.to(device), gt_img_tensor.to(device))
                #print("LPIPS distance:", lpips_distance.item())

                l2_distance = torch.norm(img_tensor - gt_img_tensor, p=2)

                tester_prompt = '' 
                text_embeddings = pipe.get_text_embedding(tester_prompt)
                det_prob = 1 - watermark_prob(img_tensor, pipe, wm_pipe, text_embeddings)

                path = os.path.join(wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png")
                save_img(path, img_tensor, pipe)
                if os.path.exists(wm_img_path):
                    os.remove(wm_img_path)

                output_dict['Detect Prob'].append(det_prob)
                output_dict['SSIM'].append(ssim_value)
                output_dict['PSNR'].append(psnr_value)
                output_dict['LPIPS'].append(lpips_distance.item())
                output_dict['L2 distance'].append(l2_distance.item())
                
                logging.info(f'Detect Prob: {det_prob} , \n SSIM {ssim_value}, PSNR {psnr_value}, LPIPS {lpips_distance.item()},  L2 distance {l2_distance.item()}')
    
    output_json = json.dumps(output_dict, indent=4)
    with open(f'{wm_path}/output.json', 'w') as f:
        f.write(output_json)




