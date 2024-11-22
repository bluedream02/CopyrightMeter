import subprocess
import os
import shutil


dataset="artist"
folder = f'/path/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)

    os.environ['MODEL_NAME']=f"/path/stable-diffusion-1-5"

    os.environ['DATA_DIR']=f"/path/{dataset}/{subfolder}"

    os.environ['OUTPUT_DIR']=f"./output_images_concept_inv_model/{dataset}/{subfolder}"

    os.environ['SAFETY_CONCEPT']=subfolder.replace("_"," ").replace("-"," ")


    if not os.path.exists(os.environ['OUTPUT_DIR']):
        command = f"bash train.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        