#!/bin/bash

cell_tiles_dir = "$1"
competition_data_csv = "$2"
external_data_csv ="$3"

competition_data_imgs = "$4"
external_data_imgs ="$5"

hpa_mask_dir = $"6"


python get_cell_comp_data.py --train_csv "$competition_data_csv" --train_data "$competition_data_imgs" --hpa_masks "$hpa_mask_dir" --cell_tiles $"hpa_mask_dir"
python get_cell_Riedel_data.py --train_csv "$external_data_csv" --train_data "$external_data_imgs"  --cell_tiles $"hpa_mask_dir"
create_folds.py
