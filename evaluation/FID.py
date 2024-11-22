import os
import shutil
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

def list_files(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def copy_images_to_new_folder(root_folder, new_folder_name):
    new_folder_path = os.path.join(root_folder, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    cnt=0

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                source_file_path = os.path.join(subdir, file)
                destination_file_path = os.path.join(new_folder_path, file)
                if source_file_path!=destination_file_path:
                    shutil.copy(source_file_path, destination_file_path+"_"+str(cnt)+".png")
                    cnt=cnt+1

    # print(new_folder_name," finished copy!")


input_dir1 = "/path/input1"
input_dir2 = "/path/input2"

list1=list_files(input_dir1)
list2=list_files(input_dir2)

path1_1=input_dir1+'/temp'
path2_1=input_dir2+'/temp'

try:
    shutil.rmtree(path1_1)
except OSError as e:
    pass
try:
    shutil.rmtree(path2_1)
except OSError as e:
    pass


copy_images_to_new_folder(input_dir1, path1_1)
copy_images_to_new_folder(input_dir2, path2_1)

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

path_list=[path1_1,path2_1]
for folder_path in path_list:
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert('RGB') 
            transformed_image = transform(image)
            save_path = os.path.join(folder_path, 'processed', filename)
            torchvision.utils.save_image(transformed_image, img_path)

try:
    fid_value = calculate_fid_given_paths([temp_dir1, temp_dir2], batch_size=50, dims=2048, cuda="cuda")
    print({"FID": fid_value})
except Exception as e:
    print(f"Error calculating FID: {e}")
finally:
    try:
        shutil.rmtree(path1_1)
    except OSError as e:
        pass
    try:
        shutil.rmtree(path2_1)
    except OSError as e:
        pass

