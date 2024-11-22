import subprocess
import os
import shutil


dataset="person"

folder = f'/path/{dataset}'

for subfolder in os.listdir(folder):
    print(subfolder)

    os.environ['DATA_DIR']=f"/path/{dataset}/{subfolder}"
    os.environ['OUTPUT_DIR']=f"./sld_model/{dataset}/{subfolder}"
    os.environ['SAFETY_CONCEPT']=subfolder.replace("-"," ")

    if not os.path.exists(os.environ['OUTPUT_DIR']):
        command = f"bash train_person.sh"
        res = subprocess.run(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
        