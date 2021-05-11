# Human Protein Atlas - Single Cell Classification

This code is my part of the solution of my team for [HPA - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification).

## Data Preparation
Download [competitiona data](https://www.kaggle.com/c/hpa-single-cell-image-classification/data), [external data](https://www.kaggle.com/alexanderriedel/hpa-public-768-excl-0-16) (thanks to [Alexander Riedel](https://www.kaggle.com/alexanderriedel)) and [cell masks for the train data](https://www.kaggle.com/its7171/hpa-mask) (thanks to [Takuya Ito](https://www.kaggle.com/its7171)).

To create cell tiles and corresponding csv files run the following command.<br/>
`$ bash preprocessing.sh <arg1> <arg2> <arg3> <arg4> <arg5> <arg6>`<br/>

#### Arguments
- `arg1`: directory where cell-tiles will be saved
- `arg2`: competition data csv file
- `arg3`: external data csv
- `arg4`: competition data images
- `arg5`: external data images
- `arg6`: path do hpa masks

## Train
To train models run the following command:<br/>
`$ bash train.sh <directory where cells were saved> `<br/>

#### Metrics Visualization
Metrics are logged to [wandb.ai](https://wandb.ai/).

## Weights preparation
To convert weight files from PyTorch Lightning to vanilla Pytorch run the following command.<br/>
```$ python convert2pytorch.py```

## Inference
You must upload the checkpoints of trained models to the kaggle dataset and create a submission notebook.
