import argparse
import pandas as pd
import numpy as np
import os
import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import zlib
import gc
import base64
from pycocotools import _mask as coco_mask
import typing as t
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required = True, help='path to train.csv')
    parser.add_argument('--train_data', type=str, required = True, help='path to train images')
    parser.add_argument('--hpa_masks', type=str, required = True, help='path to hpa masks (Tito dataset)')
    parser.add_argument('--cell_tiles', type=str, required = True, help='where cell tiles will be saved')
    
    return parser.parse_args()


args = get_args()


df = pd.read_csv(args.train_csv)

def get_cropped_cell(img, msk):
    bmask = msk.astype(int)[...,None]
    masked_img = img * bmask
    true_points = np.argwhere(bmask)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
    return cropped_arr

def get_stats(cropped_cell):
    x = (cropped_cell/255.0).reshape(-1,3).mean(0)
    x2 = ((cropped_cell/255.0)**2).reshape(-1,3).mean(0)
    return x, x2

def read_img(image_id, color, image_size=None):
    filename = f'{args.train_data}{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    return img

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
    "encode_binary_mask expects a binary mask, received dtype == %s" %
    mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
    "encode_binary_mask expects a 2d mask, received shape == %s" %
    mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode('ascii')

x_tot,x2_tot = [],[]
lbls = []
num_files = len(df)
all_cells = []

cell_mask_dir = args.hpa_masks
for idx in tqdm.tqdm(range(num_files)):
    image_id = df.iloc[idx].ID
    labels = dfs.iloc[idx].Label
    cell_mask = np.load(f'{cell_mask_dir}/{image_id}.npz')['arr_0']
    red = read_img(image_id, "red", None)
    green = read_img(image_id, "green", None)
    blue = read_img(image_id, "blue", None)
    #yellow = read_img(image_id, "yellow", train_or_test, image_size)
    stacked_image = np.transpose(np.array([blue, green, red]), (1,2,0))

    for j in range(1, np.max(cell_mask) + 1):
        bmask = (cell_mask == j)
        
        cropped_cell = get_cropped_cell(stacked_image, bmask)
        fname = f'{image_id}_{j}.jpg'
        cv2.imwrite(args.cell_tiles+fname,cropped_cell)
        x, x2 = get_stats(cropped_cell)
        x_tot.append(x)
        x2_tot.append(x2)
        all_cells.append({
            'image_id': image_id,
            'fname': fname,
            'r_mean': x[0],
            'g_mean': x[1],
            'b_mean': x[2],
            'cell_id': j,
            'image_labels': labels,
            'size1': cropped_cell.shape[0],
            'size2': cropped_cell.shape[1],
        })

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
cell_df = pd.DataFrame(all_cells)
cell_df.to_csv('cell_train_df.csv', index=False)
print('mean:',img_avr, ', std:', img_std)