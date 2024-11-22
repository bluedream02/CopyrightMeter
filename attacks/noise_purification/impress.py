import torch
import lpips
import torchvision.transforms as T
from tqdm import tqdm
from .purifier import PurifierBase
from ..utils import preprocess


class Impress(PurifierBase):
    to_pil = T.ToPILImage()

    def __init__(self):
        Impress.ARGS = {'eps': 0.1, 'iters': 40, 'clamp_min': 0, 'clamp_max': 1,
                        'lr': 0.001, 'pur_alpha': 0.5, 'noise': 0.1}

    def purify(self, model, x, x_trans, *args):
        args = self._parameter_check(args)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for name, param in model.vae.named_parameters():
            param.requires_grad = False

        x = preprocess(x).to(device)
        X_p = x.clone().detach() + (torch.randn(*x.shape) * args['noise']).to(device)
        X_p.requires_grad_(True)

        pbar = tqdm(range(args['iters']))
        criterion = torch.nn.MSELoss()
        loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
        optimizer = torch.optim.Adam([X_p], lr=args['lr'], eps=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['iters'], eta_min=1e-5)

        for _ in pbar:
            _X_p = model.vae(X_p).sample
            optimizer.zero_grad()
            lnorm_loss = criterion(_X_p, X_p)
            d = loss_fn_alex(X_p, x)
            lpips_loss = max(d - args['eps'], 0)
            loss = lnorm_loss + args['pur_alpha'] * lpips_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            X_p.data = torch.clamp(X_p, min=args['clamp_min'], max=args['clamp_max'])
            pbar.set_description(f"[Running purify]: Loss: {loss.item():.5f} | l2 dist: {lnorm_loss.item():.4} | lpips loss: {d.item():.4}")

        X_p = (X_p / 2 + 0.5).clamp(0, 1)

        return self.to_pil(X_p[0]).convert("RGB")
