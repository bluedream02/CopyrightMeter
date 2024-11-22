import subprocess
import os
import shutil


dataset="person"

folder = f'/path/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)
    os.environ['MODEL_NAME']=f"/path/exps_attn/{dataset}/{subfolder}"
    os.environ['DATA_DIR']=f"/path/{dataset}/{subfolder}"
    os.environ['OUTPUT_DIR']=f"./fmn_model/{dataset}/{subfolder}"

    # os.environ['MODEL_DIR'] = f"/data/xunaen/Text-to-image-Copyright/dataset/update_lmx/origin_data/exps_attn/person/{subfolder}"
    # os.environ['INSTANCE_DIR'] = f"{folder}/{subfolder}"
    # os.environ['DREAMBOOTH_OUTPUT_DIR'] = f"/data/xunaen/xunaen/dreambooth/erasing_dreambooth/person/{subfolder}/"
    # os.environ['PROMPT_SKS'] = f"a photo of {subfolder}"
    if not os.path.exists(f"./fmn_model/{dataset}/{subfolder}"):
        command = f"bash train_person.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        