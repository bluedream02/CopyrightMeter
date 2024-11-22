import os
import shutil
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import lpips
from skimage.metrics import structural_similarity as ssim
import json



# Metrics Calculation Functions
def SSIM(path1, path2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return ssim(gray_image1, gray_image2)


def LPIPS(path1, path2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image1 = transform(Image.open(path1).convert('RGB')).unsqueeze(0)
    image2 = transform(Image.open(path2).convert('RGB')).unsqueeze(0)

    lpips_model = lpips.LPIPS(net='vgg')
    return lpips_model(image1, image2).item()


def PSNR(path1, path2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    mse = np.mean((gray_image1 - gray_image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def VIFP(path1, path2):
    def calculate_vif(image1, image2, sigma_nsq=0.4):
        mu1 = cv2.GaussianBlur(image1, (11, 11), sigma_nsq)
        mu2 = cv2.GaussianBlur(image2, (11, 11), sigma_nsq)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(image1 * image1, (11, 11), sigma_nsq) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(image2 * image2, (11, 11), sigma_nsq) - mu2_sq
        sigma12 = cv2.GaussianBlur(image1 * image2, (11, 11), sigma_nsq) - mu1_mu2

        num = 2 * mu1_mu2 + 0.01
        den = mu1_sq + mu2_sq + 0.01
        return np.mean(num / den)

    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)


    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return calculate_vif(gray_image1, gray_image2)


# Helper and Main Functions
def list_files(root_dir, extensions=(".jpg", ".png")):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                file_list.append(os.path.join(root, file))
    return file_list


def calculate_metrics(input_dir1, input_dir2):
    list1 = list_files(input_dir1)
    list2 = list_files(input_dir2)

    SSIM_list, LPIPS_list, PSNR_list, VIFP_list = [], [], [], []

    for file1 in list1:
        for file2 in list2:
            if os.path.basename(file1) == os.path.basename(file2) and os.path.dirname(file1).split('/')[-1] == os.path.dirname(file2).split('/')[-1]:
                print(f"Comparing {file1} with {file2}")
                try:
                    SSIM_list.append(SSIM(file1, file2))
                except Exception as e:
                    print(f"Error calculating SSIM for {file1} and {file2}: {e}")
                try:
                    LPIPS_list.append(LPIPS(file1, file2))
                except Exception as e:
                    print(f"Error calculating LPIPS for {file1} and {file2}: {e}")
                try:
                    PSNR_list.append(PSNR(file1, file2))
                except Exception as e:
                    print(f"Error calculating PSNR for {file1} and {file2}: {e}")
                try:
                    VIFP_list.append(VIFP(file1, file2))
                except Exception as e:
                    print(f"Error calculating VIFP for {file1} and {file2}: {e}")

    return {
        "SSIM": {"list": SSIM_list, "avg": np.mean(SSIM_list), "std": np.std(SSIM_list)},
        "LPIPS": {"list": LPIPS_list, "avg": np.mean(LPIPS_list), "std": np.std(LPIPS_list)},
        "PSNR": {"list": PSNR_list, "avg": np.mean(PSNR_list), "std": np.std(PSNR_list)},
        "VIFP": {"list": VIFP_list, "avg": np.mean(VIFP_list), "std": np.std(VIFP_list)},
    }


def save_results(data, output_dir, output_file="evaluation.json"):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, output_file)
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Results saved to {json_path}")



if __name__ == "__main__":
    input_dir1 = "/path/input"
    input_dir2 = "/path/output"
    output_dir = os.path.join(input_dir1, "_result")

    results = calculate_metrics(input_dir1, input_dir2)
    save_results(results, output_dir)