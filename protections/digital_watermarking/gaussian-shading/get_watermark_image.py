import subprocess
import os
import shutil

dataset="artist"
dataset="person"
dataset="CustomConcept"
folder = f'/path/input/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    if dataset=="artist":
        pre="a painting in the style of "
    elif dataset=="person":
        pre="a photo of "
    elif dataset=="CustomConcept":
        pre="a photo of a "

    os.environ['ARTIST']=f"{pre}{subfolder.replace('-',' ').replace('_',' ')}"
    os.environ['OUTPUT_DIR']=f"/path/outputs/{dataset}/{subfolder}"
    if not os.path.exists(f"/path/outputs/{dataset}/{subfolder}"):
        command = f"bash get_watermark_image.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        