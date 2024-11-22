import torch
import pytorch_ssim
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def purify_imagenet(x, diffusion, model,  max_iter, device):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = transforms.Compose(
    [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )
    transform_diff_to_raw = transforms.Compose(
    [
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
    ]
    )
    x_adv = transform_raw_to_diff(x).to(device)
    print('x_adv')
    print(x_adv.shape)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256

    t_steps = torch.ones(x_adv.shape[0], device=device).long()
    t_steps = t_steps * (45-1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        print(f'cond_fn{t}')
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            
            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            
            selected = pytorch_ssim.ssim(x_in, x_adv_t)
            scale = diffusion.compute_scale(x_in,t, 4.*2/255. / 3. / 1000)
            
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale


    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):            
            adv_sample = diffusion.q_sample(xt_reverse,t_steps)
            sample_fn = diffusion.p_sample_loop if not False else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                    model,
                    shape,
                    num_purifysteps = 45,
                    noise = adv_sample,
                    clip_denoised=True,
                    cond_fn = cond_fn if True else None,
                    model_kwargs=model_kwargs,
                )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224], mode="bilinear") # transfrom size 256 -> 224
            images.append(x_pur)

    return images


class diffusion_pur:
    def __init__(self,model_path):
        #self.input_image_path = input_image_path
        #self.image = Image.open(input_image_path)
        self.max_iter = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        num_channels=256,
        num_res_blocks=2,
        channel_mult='',
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions='32,16,8',
        dropout=0.0,
        diffusion_steps=1000,
        noise_schedule='linear',
        timestep_respacing='250',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,)
        self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
        self.model.to(self.device)
        self.model.convert_to_fp16()
        self.model.eval()
        self.transform = transforms.Compose([
        # transforms.Resize(256),        # 调整大小为 256x256
        # transforms.CenterCrop(224),    # 中心裁剪为 224x224
        transforms.Resize((224, 224)),
        transforms.ToTensor()          # 转换为 Tensor
        ])

    def generate(self,input_image_path):
        image = Image.open(input_image_path)
        transformed_image = self.transform(image)
        testLoader = transformed_image.reshape((-1,3,224,224))
        for i, x in enumerate(testLoader):
            x = x.reshape((-1,3,224,224))
            print("Epoch {}".format(i))
            print('x')
            # purify natural image
            x_nat_pur_list_list = []
            for j in range(5):
                x_nat_pur_list = purify_imagenet(x=x, diffusion=self.diffusion, model=self.model, max_iter=self.max_iter,device=self.device)
                x_nat_pur_list_list.append(x_nat_pur_list)
        for i in x_nat_pur_list_list[-1]:
            i = i.reshape((3,224,224))
            i.detach()
        return i#返回的是图片tensor  


def process_folder(input_folder, output_folder, model_path):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    diffusion_purify = diffusion_pur(model_path=model_path)

    # 处理输入文件夹中的每张图片
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, filename)
                # 生成处理后的图像
                image = diffusion_purify.generate(input_image_path)
                # 确保输出文件夹结构与输入文件夹相同
                relative_path = os.path.relpath(input_image_path, input_folder)
                output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                # 保存处理后的图像到输出文件夹
                output_image_path = os.path.join(output_subfolder, filename)
                transformed_image_pil = TF.to_pil_image(image)
                transformed_image_pil.save(output_image_path)


if __name__ == '__main__':
    input_folder = "/path/input"
    output_folder = "/path/output"
    model_path = "/path/256x256_diffusion_uncond.pt"
    process_folder(input_folder, output_folder, model_path)
