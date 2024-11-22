import os
import subprocess
from pathlib import Path
from PIL import Image


class Mist:
    def __init__(self, mist_v3_folder, input_image_path, output_image_path, 
                 epsilon=16, steps=100, input_size=512, mode=2):
        """
        Mist attack for a single image.
        :param mist_v3_folder: Path to the folder containing mist_v3.py
        :param input_image_path: Path to the input image
        :param output_image_path: Path to save the output image
        :param epsilon: Strength of the Mist attack
        :param steps: Number of steps for the attack
        :param input_size: Size of the input image for the attack
        :param mode: Mode of the Mist attack
        """
        if not mist_v3_folder:
            raise ValueError("Please provide the path to the folder containing mist_v3.py")
        
        self.mist_v3_folder = Path(mist_v3_folder)
        self.input_image_path = Path(input_image_path)
        self.output_image_path = Path(output_image_path)
        self.epsilon = epsilon
        self.steps = steps
        self.input_size = input_size
        self.mode = mode
        
        # Ensure the input image exists
        if not self.input_image_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.input_image_path}")

        # Create output directory if it doesn't exist
        self.output_image_path.parent.mkdir(parents=True, exist_ok=True)
    
    def generate(self):
        """
        Run the Mist attack on the input image and save the output image.
        """
        command = [
            "python", "mist_v3.py",
            "--input_image_path", str(self.input_image_path),
            "--output_name", self.output_image_path.stem,
            "--output_dir", str(self.output_image_path.parent),
            "--epsilon", str(self.epsilon),
            "--steps", str(self.steps),
            "--input_size", str(self.input_size),
            "--mode", str(self.mode)
        ]

        try:
            # Change to the folder containing mist_v3.py and execute the command
            os.chdir(self.mist_v3_folder)
            subprocess.run(command, check=True)
            print(f"Output saved to {self.output_image_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during Mist attack: {e}")
        finally:
            # Restore the original working directory
            os.chdir(Path.cwd())


# Example usage:
# mist = Mist("/path/to/mist-main", "test/sample.png", "output/misted_sample.png")
# mist.generate()
