import argparse
import subprocess

class MistUse:
    def __init__(self, input_image_path="test/sample.png", output_name="misted_sample", output_dir="vangogh",
                 input_dir_path=None, epsilon=16, steps=100, input_size=512, block_num=1, mode=2, rate=1,
                 mask=False, mask_path="test/processed_mask.png", non_resize=False):
        
        self.command = ['python', 'mist_v3.py']  # 命令行基本命令
        self.args = self._parse_args(input_image_path, output_name, output_dir, input_dir_path, epsilon, steps,
                                      input_size, block_num, mode, rate, mask, mask_path, non_resize)
    
    def _parse_args(self, input_image_path, output_name, output_dir, input_dir_path, epsilon, steps, input_size,
                    block_num, mode, rate, mask, mask_path, non_resize):
        parser = argparse.ArgumentParser(description="Configs for Mist V1.2")
        parser.add_argument(
            "-img",
            "--input_image_path",
            type=str,
            default=input_image_path,
            help="path of input image",
        )
        parser.add_argument(
            "--output_name",
            type=str,
            default=output_name,
            help="path of saved image",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=output_dir,
            help="path of output dir",
        )
        parser.add_argument(
            "-inp",
            "--input_dir_path",
            type=str,
            default=input_dir_path,
            help="Path of the dir of images to be processed.",
        )
        parser.add_argument(
            "-e",
            "--epsilon",
            type=int,
            default=epsilon,
            help=(
                "The strength of Mist"
            ),
        )
        parser.add_argument(
            "-s",
            "--steps",
            type=int,
            default=steps,
            help=(
                "The step of Mist"
            ),
        )
        parser.add_argument(
            "-in_size",
            "--input_size",
            type=int,
            default=input_size,
            help=(
                "The input_size of Mist"
            ),
        )
        parser.add_argument(
            "-b",
            "--block_num",
            type=int,
            default=block_num,
            help=(
                "The number of partitioned blocks"
            ),
        )
        parser.add_argument(
            "--mode",
            type=int,
            default=mode,
            help=(
                "The mode of MIST."
            ),
        )
        parser.add_argument(
            "--rate",
            type=int,
            default=rate,
            help=(
                "The fused weight under the fused mode."
            ),
        )
        parser.add_argument(
            "--mask",
            default=mask,
            action="store_true",
            help=(
                "Whether to mask certain region of Mist or not. Work only when input_dir_path is None. "
            ),
        )
        parser.add_argument(
            "--mask_path",
            type=str,
            default=mask_path,
            help="Path of the mask.",
        )
        parser.add_argument(
            "--non_resize",
            default=non_resize,
            action="store_true",
            help=(
                "Whether to keep the original shape of the image or not."
            ),
        )
        return parser.parse_args()

    def run(self):
        try:
            args = vars(self.args)
            for arg, value in args.items():
                if arg not in ['mask', 'non_resize']:
                    self.command.extend([f"--{arg}", str(value)])
                elif value:
                    self.command.append(f"--{arg}")
            subprocess.run(self.command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error:", e)

mist = MistUse(input_dir_path="test/vangogh", output_dir="vangogh")
mist.run()
