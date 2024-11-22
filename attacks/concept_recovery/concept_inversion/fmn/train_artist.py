import subprocess
import os
import shutil

dataset="artist"

folder = f'/data/xunaen/Text-to-image-Copyright/dataset/update_lmx/origin_data/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    os.environ['MODEL_NAME']=f"/path/exps_attn/{dataset}/{subfolder}"
    os.environ['DATA_DIR']=f"/path/{dataset}/{subfolder}"
    os.environ['OUTPUT_DIR']=f"./fmn_model/{dataset}/{subfolder}"
    if not os.path.exists(f"./fmn_model/{dataset}/{subfolder}"):
        command = f"bash train_artist.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        