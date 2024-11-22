import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
import requests
from io import BytesIO
#from Utils.adversarial_generator_tools import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image
from typing import Union, List, Optional, Callable
from tqdm import tqdm
import numpy as np
from .generator import GeneratorBase
from ..utils import prepare_mask_and_masked_image, recover_image, prepare_image

class DiffusionAttack:
    def __init__(self, clamp_min=-1, clamp_max=1, eps=16, step_size=1, 
        iters=200, size=512, eta=1, grad_reps=10, guidance_scale=7.5, num_inference_steps=4, mode="l2", target_image=None):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = eps
        self.step_size = step_size
        self.iters = iters
        self.size = size
        self.eta = eta
        self.grad_reps = grad_reps
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.mode = mode
        self.target_image = target_image
        
    def attack_forward(self, 
        model,
        prompt: Union[str, List[str]],
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
    ):

        text_inputs = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = model.text_encoder(text_input_ids.to(model.device))[0]
        # 创建一个空的无条件输入，用于后续的文本嵌入拼接
        uncond_tokens = [""]
        max_length = text_input_ids.shape[-1]
        uncond_input = model.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        # 拼接无条件嵌入和条件文本嵌入  
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        seq_len = uncond_embeddings.shape[1]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # 从计算图中分离文本嵌入，这样在后续的梯度计算中不会考虑它们
        text_embeddings = text_embeddings.detach()
        # 获取VAE模型配置中的潜在通道数
        num_channels_latents = model.vae.config.latent_channels
        # 初始化潜在向量
        latents_shape = (1 , num_channels_latents, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=model.device, dtype=text_embeddings.dtype)

        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2)
        # 编码mask图像并获取其潜在分布样本
        masked_image_latents = model.vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)
        # 根据初始噪声sigma对潜在向量进行缩放
        latents = latents * model.scheduler.init_noise_sigma
        
        model.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = model.scheduler.timesteps.to(model.device)

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = model.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        latents = 1 / 0.18215 * latents
        image = model.vae.decode(latents).sample
        return image

        
    def compute_grad(self, inpaint_model, cur_mask, cur_masked_image, prompt, target_image, **kwargs):
        torch.set_grad_enabled(True)
        cur_mask = cur_mask.clone()
        cur_masked_image = cur_masked_image.clone()
        cur_mask.requires_grad = False
        cur_masked_image.requires_grad_()
        image_nat = self.attack_forward(inpaint_model,mask=cur_mask,
                                masked_image=cur_masked_image,
                                prompt=prompt,
                                **kwargs)
        
        loss = (image_nat - target_image).norm(p=2)
        grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)
            
        return grad, loss.item(), image_nat.data.cpu()

    def super_l2(self, inpaint_model, cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps = 5, target_image = 0, **kwargs):
        X_adv = X.clone()
        iterator = tqdm(range(iters))
        for i in iterator:

            all_grads = []
            losses = []
            for i in range(grad_reps):
                c_grad, loss, last_image = self.compute_grad(inpaint_model, cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
                all_grads.append(c_grad)
                losses.append(loss)
            grad = torch.stack(all_grads).mean(0)
            
            iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

            l = len(X.shape) - 1
            grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
            grad_normalized = grad.detach() / (grad_norm + 1e-10)

            # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
            actual_step_size = step_size
            X_adv = X_adv - grad_normalized * actual_step_size

            d_x = X_adv - X.detach()
            d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
            X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)        
        
        torch.cuda.empty_cache()
        return X_adv, last_image

    def super_linf(self, inpaint_model, cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps = 5, target_image = 0, **kwargs):
        X_adv = X.clone()
        iterator = tqdm(range(iters))
        for i in iterator:

            all_grads = []
            losses = []
            for i in range(grad_reps):
                c_grad, loss, last_image = self.compute_grad(inpaint_model, cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
                all_grads.append(c_grad)
                losses.append(loss)
            grad = torch.stack(all_grads).mean(0)
            
            iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
            # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
            actual_step_size = step_size
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        torch.cuda.empty_cache()
        return X_adv, last_image

    def inpaint_diffusion_attack(self, image, mask_image, inpaint_model, prompt=""):

        """
            基于inpaint模型进行扩散攻击
        """
        if self.mode not in ["l2", "linf"]:
            print("wrong setting")
            return None
        image = image.convert('RGB').resize((self.size,self.size))
        mask_image = mask_image.convert("RGB")
        mask_image = ImageOps.invert(mask_image).resize((self.size,self.size))
        # 看是否指定了目标图像，如果没有指定，则使用默认的目标图像，否则就使用传入的图像
        if target_image is None:
            target_url = "https://i.pinimg.com/originals/18/37/aa/1837aa6f2c357badf0f588916f3980bd.png"
            response = requests.get(target_url)
            target_image = Image.open(BytesIO(response.content)).convert("RGB")
            target_image = target_image.resize((self.size, self.size))
        else:
            target_image = target_image.convert("RGB").resize((self.size, self.size))
        prompt = prompt
        SEED = 786349
        torch.manual_seed(SEED)
        strength = 0.7
        guidance_scale = guidance_scale
        num_inference_steps = num_inference_steps
        cur_mask, cur_masked_image = prepare_mask_and_masked_image(image, mask_image)
        cur_mask = cur_mask.half().cuda()
        cur_masked_image = cur_masked_image.half().cuda()
        target_image_tensor = prepare_image(target_image)
        target_image_tensor = 0*target_image_tensor.cuda() # we can either attack towards a target image or simply the zero tensor
        if self.mode == "l2":
            result, last_image= self.super_l2(inpaint_model, cur_mask, cur_masked_image,
                prompt=prompt,
                target_image=target_image_tensor,
                eps=self.eps,
                step_size=self.step_size,
                iters=self.iters,
                clamp_min = self.clamp_min,
                clamp_max = self.clamp_max,
                eta=self.eta,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                grad_reps=self.grad_reps
                )
        else:
            result, last_image= self.super_linf(cur_mask, cur_masked_image,
                prompt=prompt,
                target_image=target_image_tensor,
                eps=self.eps,
                step_size=self.step_size,
                iters=self.iters,
                clamp_min = self.clamp_min,
                clamp_max = self.clamp_max,
                height = self.size,
                width = self.size,
                eta=self.eta,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                )
        adv_image = (result / 2 + 0.5).clamp(0, 1)
        adv_image = T.ToPILImage()(adv_image[0]).convert("RGB")
        adv_image = recover_image(adv_image, image, mask_image, background=True)
        return adv_image

    def generate(self, image, mask_image, model, prompt):
        return self.inpaint_diffusion_attack(image, mask_image, model, prompt)