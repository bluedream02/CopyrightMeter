#!/bin/bash

dataset="WikiArt"

dir_path="/path/input/$dataset"
dir_path2="/path/output"

for folder in $(ls -d $dir_path/*/); do
  folder_name=$(basename $folder)
  sed -i "s|instance_data_dir: .*|instance_data_dir: $folder|" configs/ti.yaml
  sed -i "s|output_dir: .*|output_dir: $dir_path2/exps_ti/$dataset/$folder_name|" configs/ti.yaml
  python run.py configs/ti.yaml
done