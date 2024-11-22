from main.wmattacker import *

attackers = {
        'random_crop_0.6': RandomCropAttacker(percent=0.6),
        'random_drop_0.6': RandomDropAttacker(percent=0.6),
        'median_filter_7': MedianFilterAttacker(size=7),
        'gaussian_noise_mean0_sigma0.05': GaussianNoiseAttacker(mean=0, sigma=0.05),
        'salt_pepper_noise_0.05': SaltPepperNoiseAttacker(prob=0.05),
        'resize_restore_25': ResizeRestoreAttacker(scale_percent=25),
        'cheng2020-anchor': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
        'bmshj2018-factorized': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        'rotate_90': RotateAttacker(degree=90),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'contrast_0.5': ContrastAttacker(contrast=0.5),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
        'bm3d': BM3DAttacker(),
        }


dataset='artist'
folder = f'/path/{dataset}'


for subfolder in os.listdir(folder):
    original_directory = f'/path/watermarked_image/{dataset}/{subfolder}/'
    
    png_files = [file for file in os.listdir(original_directory) if file.endswith('.png')]

    for file_name in png_files:
        input=os.path.join(original_directory, file_name)
        for idx, (attacker_name, attacker) in enumerate(attackers.items(),start=14):
            output=f'./attack_images/{dataset}/{idx}_{attacker_name}/{subfolder}/'
            os.makedirs(output, exist_ok=True)
            output=os.path.join(output, file_name)
            attackers[attacker_name].attack([input], [output], multi=True)