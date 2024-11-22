import torch
import lpips
import torch.nn as nn
import torchvision.transforms as T

from tqdm import tqdm
from .generator import GeneratorBase
from ..utils import preprocess


class Glaze(GeneratorBase):
    to_pil = T.ToPILImage()

    def __init__(self):
        Glaze.ARGS = {'p': 0.1, 'alpha': 0.1, 'iters': 200, 'lr': 0.00005}
        pass

    def generate(self, model, x, x_trans, *args):
        for name, param in model.vae.encoder.named_parameters():
            param.requires_grad = False
        args = self._parameter_check(args)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        x_trans = x_trans.resize(x.size)
        x = preprocess(x).to(device)
        x_t = preprocess(x_trans).to(device)

        delta = (torch.rand(*x.shape) * 2 * args['p'] - args['p']).to(device)
        pbar = tqdm(range(args['iters']))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam([delta], lr=args['lr'])
        loss_fn_alex = lpips.LPIPS(net='vgg').to(device)

        for _ in pbar:
            delta.requires_grad_(True)
            x_adv = x + delta
            x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
            x_emb = model.vae.encode(x_adv).latent_dist.sample()
            x_trans_emb = model.vae.encode(x_t).latent_dist.sample()

            optimizer.zero_grad()
            d = loss_fn_alex(x, x_adv)
            sim_loss = args['alpha'] * max(d-args['p'], 0)
            loss = criterion(x_emb, x_trans_emb) + sim_loss

            loss.backward()
            optimizer.step()

            # pbar.set_description(f"[Running glaze]: Loss {loss.item():.5f}  \
            #                      | sim loss {args['alpha'] * max(d.item()-args['p'], 0):.5f} \
            #                      | dist {d.item():.5f}")
        x_adv = x + delta
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)

        x_adv = (x_adv / 2 + 0.5).clamp(0, 1)
        return self.to_pil(x_adv[0]).convert("RGB")
