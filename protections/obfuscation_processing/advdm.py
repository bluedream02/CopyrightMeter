import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
from .generator import GeneratorBase
from einops import rearrange


class AdvDM(GeneratorBase):
    def __init__(self):
        AdvDM.ARGS = {'iters': 50, 'ls': 0.01, 'eps': 0.03}
        pass

    def generate(self, model, x, x_trans, *args):
        args = self._parameter_check(args)
        trans = transforms.Compose([transforms.ToTensor()])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x_trans = x_trans.resize(x.size)
        img = np.array(x).astype(np.float16) / 127.5 - 1.0
        img = img[:, :, :3]
        tar_img = np.array(x_trans).astype(np.float16) / 127.5 - 1.0
        tar_img = tar_img[:, :, :3]

        data_source = torch.zeros([1, 3, img.shape[0], img.shape[1]]).to(device)
        data_source[0] = trans(img).to(device)
        target = torch.zeros([1, 3, tar_img.shape[0], tar_img.shape[1]]).to(device)
        target[0] = trans(tar_img).to(device)

        vae = model.vae
        vae.requires_grad_(False)
        loss_fn = nn.MSELoss(reduction="sum")
        delta = torch.zeros_like(data_source)
        delta = nn.Parameter(delta)

        target_tensor = vae.encode(target).latent_dist.sample()
        for _ in tqdm(range(args['iters'])):
            delta.requires_grad_()
            output = vae.encode(data_source + delta).latent_dist.sample()

            loss = loss_fn(output, target_tensor)
            loss.backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * args['ls']
            delta.data = torch.clamp(delta.data, -args['eps'], args['eps'])
            delta.data = torch.clamp(delta.data + data_source.data, -1.0, 1.0)

            delta.grad.data.zero_()

        output = (delta + data_source).data[0]
        save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
        grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
        return Image.fromarray(grid_adv.astype(np.uint8))