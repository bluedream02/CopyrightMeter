import os
import subprocess

class AntiDreamBoothUse:
    def __init__(self, antidream_path, script_name):
        self.antidream_path = antidream_path
        self.script_name = script_name

    def generate(self):
        try:
            os.chdir(self.antidream_path)  # 进入 antidream 文件夹
            if self.script_name == "attack_with_aspl":
                subprocess.run(["bash", "scripts/attack_with_aspl.sh"], check=True)
            elif self.script_name == "attack_with_targeted_aspl":
                subprocess.run(["bash", "scripts/attack_with_targeted_aspl.sh"], check=True)
            elif self.script_name == "attack_with_ensemble_aspl":
                subprocess.run(["bash", "scripts/attack_with_ensemble_aspl.sh"], check=True)
            elif self.script_name == "attack_with_fsmg":
                subprocess.run(["bash", "scripts/attack_with_fsmg.sh"], check=True)
            elif self.script_name == "attack_with_targeted_fsmg":
                subprocess.run(["bash", "scripts/attack_with_targeted_fsmg.sh"], check=True)
            elif self.script_name == "attack_with_ensemble_fsmg":
                subprocess.run(["bash", "scripts/attack_with_ensemble_fsmg.sh"], check=True)
            elif self.script_name == "train_dreambooth_alone":
                subprocess.run(["bash", "scripts/train_dreambooth_alone.sh"], check=True)
            else:
                print("Invalid script name!")
        except subprocess.CalledProcessError as e:
            print("Error:", e)
'''
# Example usage:
if __name__ == "__main__":
    anti_dreambooth_use = AntiDreamBoothUse("/root/autodl-tmp/Anti-DreamBooth-main", "attack_with_aspl")
    anti_dreambooth_use.generate()
'''
