import os
import cv2
import torch
from torch.utils.data.dataset import Dataset

class CellDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.df = csv_file
        self.transforms = transform           
        self.cell_types = self.df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']].values
        self.img_ids = self.df['image_id'].values
        self.cell_ids = self.df['cell_id'].values

    def __len__(self):
        return len(self.img_ids)
        # return 100

    def get_image(self, index):

        image_id = self.img_ids[index]
        cell_id = self.cell_ids[index]
        
        img_path = os.path.join(self.data_dir, 'cells', image_id + '_' + str(cell_id) + '.jpg')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)
        img = img['image']

        return img

    def __getitem__(self, index):

        x = self.get_image(index)
        y = self.cell_types[index]
        y = torch.from_numpy(y).float()
        return x, y

        