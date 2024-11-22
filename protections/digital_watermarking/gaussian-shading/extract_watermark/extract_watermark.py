import subprocess
import os
import shutil

for dataset in ['WikiArt','person','CustomConcept']:
    folder = f'/path/input/{dataset}'

    for subfolder in os.listdir(folder):
        print(subfolder)
        os.environ['IMAGE_DIR']=f"./attack_images/{dataset}/0_jpeg/{subfolder}/"
        command = f"bash extract_watermark.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        