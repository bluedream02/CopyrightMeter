import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
import requests
from io import BytesIO
#from Utils.adversarial_generator_tools import preprocess, prepare_mask_and_masked_image, recover_image
from tqdm import tqdm
from PIL import Image, ImageOps
from .generator import GeneratorBase
from ..utils import preprocess, prepare_mask_and_masked_image, recover_image


def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None, mode="img2img", target=None):
        """
            内部函数，定义投射梯度下降法(PGD)攻击
        """
        # 判断设定，设定错误直接退出
        if mode not in ["img2img", "inpaint"]:
            print("wrong setting")
            return None
        if mode == "inpaint" and target is None:
            print("wrong setting")
            return None
        # 初始化对抗样本
        X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
        pbar = tqdm(range(iters))
        for i in pbar:
            actual_step_size = step_size - (step_size - step_size / 100) / iters * i  
            X_adv.requires_grad_(True)
            # 更新损失
            if mode == "img2img":
                loss = (model(X_adv).latent_dist.mean).norm()
            elif mode == "inpaint":
                loss = (model(X_adv).latent_dist.mean - target).norm()
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
            grad, = torch.autograd.grad(loss, [X_adv])
            # 梯度下降法更新对抗样本
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            # 把更新后的对抗样本裁剪到原始图像的eps范围内
            X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
            X_adv.grad = None    
            if mask is not None:
                X_adv.data *= mask
        return X_adv

class EncoderAttackImg2Img:
    def __init__(self):
        pass

    def img2img_encoder_attack(self, image, img2img_model, clamp_min=-1, clamp_max=1, eps=0.06, step_size=0.02, iters=100, size=512, mode="test"):
        """
            基于img2img模型进行编码器攻击
        """
        if mode not in ["test", "use"]:
            return None
        original_size = image.size
        resize = T.transforms.Resize(size)
        center_crop = T.transforms.CenterCrop(size)
        init_image = center_crop(resize(image))
        with torch.autocast('cuda'):
            processed_image = preprocess(init_image).half().cuda()
            adv_image = pgd(processed_image, 
                        model=img2img_model.vae.encode, 
                        clamp_min=clamp_min, 
                        clamp_max=clamp_max,
                        eps=eps, # The higher, the less imperceptible the attack is 
                        step_size=step_size, # Set smaller than eps
                        iters=iters, # The higher, the stronger your attack will be
                        mode="img2img"
                    )
            # convert pixels back to [0,1] range
            adv_image = (adv_image / 2 + 0.5).clamp(0, 1)
        adv_image = T.ToPILImage()(adv_image[0]).convert("RGB")
        if mode == "test":
            return adv_image
        adv_image = adv_image.resize(original_size, Image.LANCZOS)  
        return adv_image


    def generate(self, image, img2img_model, clamp_min=-1, clamp_max=1, eps=0.06, step_size=0.02, iters=100, size=512, mode="test"):
        """
            生成编码器攻击图像
        """
        return self.img2img_encoder_attack(image, img2img_model, clamp_min, clamp_max, eps, step_size, iters, size, mode)

class EncoderAttackInpaint:
    def __init__(self):
        pass
    
    def inpaint_encoder_attack(self, image, mask_image, inpaint_model, clamp_min=-1, clamp_max=1, eps=0.06, step_size=0.01, 
        iters=500, size=512, mode="test", target_image=None):
        """
            基于inpaint模型进行编码器攻击
        """
        image = image.convert('RGB').resize((size,size))
        mask_image = mask_image.convert("RGB")
        mask_image = ImageOps.invert(mask_image).resize((size,size))
        # 看是否指定了目标图像，如果没有指定，则使用默认的目标图像，否则就使用传入的图像
        if target_image is None:
            target_url = "https://bostonglobe-prod.cdn.arcpublishing.com/resizer/2-ZvyQ3aRNl_VNo7ja51BM5-Kpk=/960x0/cloudfront-us-east-1.images.arcpublishing.com/bostonglobe/CZOXE32LQQX5UNAB42AOA3SUY4.jpg"
            response = requests.get(target_url)
            target_image = Image.open(BytesIO(response.content)).convert("RGB")
            target_image = target_image.resize((size, size))
        else:
            target_image = target_image.convert("RGB").resize((size, size))
        with torch.autocast('cuda'):
            mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
            masked_image = masked_image.half().cuda()
            mask = mask.half().cuda()
            
            target = inpaint_model.vae.encode(preprocess(target_image).half().cuda()).latent_dist.mean
            adv_image = pgd(masked_image, 
                        target = target,
                        model=inpaint_model.vae.encode, 
                        clamp_min=clamp_min, 
                        clamp_max=clamp_max,
                        eps=eps, 
                        step_size=step_size, 
                        iters=iters, #Increase for better attack (but slower)
                        mask=1-mask,
                        mode="inpaint"
                    )
        # 还原图像
        adv_image = (adv_image / 2 + 0.5).clamp(0, 1)
        adv_image = T.ToPILImage()(adv_image[0]).convert("RGB")
        adv_image = recover_image(adv_image, image, mask_image, background=True)
        return adv_image
    
    def generate(self, image, mask_image, inpaint_model, clamp_min=-1, clamp_max=1, eps=0.06, step_size=0.01, iters=500, size=512, mode="test", target_image=None):
        return self.inpaint_encoder_attack(image, mask_image, inpaint_model, clamp_min, clamp_max, eps, step_size, iters, size, mode, target_image)