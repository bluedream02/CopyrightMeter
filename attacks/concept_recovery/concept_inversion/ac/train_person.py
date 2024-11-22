import subprocess
import os
import shutil


dataset="person"

folder = f'/path/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    os.environ['DATA_DIR']=f"/path/{dataset}/{subfolder}"
    os.environ['OUTPUT_DIR']=f"./ac_model/{dataset}/{subfolder}"

    os.environ['ESD_CKPT']=f"/path/concept-ablation/diffusers/logs_ablation/{dataset}/{subfolder}/delta.bin"

    if not os.path.exists(f"./ac_model/{dataset}/{subfolder}"):
        command = f"bash train_artist.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        