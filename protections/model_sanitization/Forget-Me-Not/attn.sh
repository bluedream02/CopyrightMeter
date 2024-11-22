#!/bin/bash

dataset="WikiArt"

dir_path="/path/input/$dataset"
dir_path2="/path/output"


for folder in $(ls -d $dir_path/*/); do
  folder_name=$(basename $folder)
  cp configs/attn.yaml configs/attn.yaml.bak
  new_multi_concept="    - [${folder_name}, style]"
  sed -i "/multi_concept:/!b;n;c\\$new_multi_concept" configs/attn.yaml
  sed -i "s|output_dir: .*|output_dir: $dir_path2/exps_attn/$dataset/$folder_name|" configs/attn.yaml
  python run.py configs/attn.yaml
done
