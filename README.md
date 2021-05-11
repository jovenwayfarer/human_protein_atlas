# Human Protein Atlas - Single Cell Classification

This is code for my part of the solution of my team for [HPA - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification).

## Data Preparation
Download [Competitiona data](https://www.kaggle.com/c/hpa-single-cell-image-classification/data), [external data](https://www.kaggle.com/alexanderriedel/hpa-public-768-excl-0-16) (thanks to [Alexander Riedel](https://www.kaggle.com/alexanderriedel) and [cell masks for the train data](https://www.kaggle.com/its7171/hpa-mask) (thanks to [Takuya Ito](https://www.kaggle.com/its7171)).

You can download via a browser or use the following commands (kaggle api required).<br/>
```$ kaggle datasets download -d its7171/hpa-mask ```<br/>
```$ kaggle datasets download -d alexanderriedel/hpa-public-768-excl-0-16 ```<br/>
```$ kaggle competitions download -c hpa-single-cell-image-classification ```<br/>

To create cell tiles and corresponding csv files run the following commands.<br/>
```$ python get_cells_comp_data --train_csv <csv for competition data> --train_data <path to compet. train images> --hpa_masks <path to masks> --cell_tiles <directory where cell tiles will be saved>```<br/>

```$ python get_cells_Riedel_data --train_csv <csv for external data> --train_data <path to external train images> --cell_tiles <directory where cell tiles will be saved>```<br/>

```$ python create_folds.py```<br/>  

## Train
To train models run the following command:<br/>

#### Metrics Visualization
Metrics are logged to [wandb.ai](https://wandb.ai/).

## Weights preparation
To convert weight files from PyTorch Lightning to vanilla Pytorch run the following command.<br/>
```$ python convert2pytorch.py```
