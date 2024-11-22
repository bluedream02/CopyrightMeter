import subprocess
import os
import shutil

dataset="WikiArt"

folder = f'/path/input/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    # if dataset=="artist":
    #     pre="a painting in the style of "
    # elif dataset=="person":
    #     pre="a photo of "
    # elif dataset=="CustomConcept":
    #     pre="a photo of a "

    artist=f"{subfolder.replace('-',' ').replace('_',' ')}"
    os.environ['ARTIST']=artist
    os.environ['DATASET']=dataset
    if not os.path.isfile(f'output_images/{dataset}/{artist}/'):
        command = f"bash gen.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
            