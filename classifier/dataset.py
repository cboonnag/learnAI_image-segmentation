import os 
import numpy as np
import pandas as pd 

from PIL import Image
import torch 
from torch.utils.data import Dataset

class CatDogMiniDataset(Dataset):
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir 
        self.target = pd.read_csv(image_dir + "annotations.csv")
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.target.iloc[index, 0]) # filename from csv file
        image = np.array(Image.open(img_path).convert("RGB"), dtype = 'float32')
        target_label = torch.tensor( int (self.target.iloc[index, 1])) # filename from csv file

        if self.transform :
            image = self.transform(image = image)['image']

        return (image, target_label)


if __name__ == "__main__":
    image_dir = 'data/train/'
    dataset = CatDogMiniDataset(image_dir)

    img, target_label = dataset[2]
    print(img)
    print(img.shape)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose([
                        A.Resize(height = 32, width = 32),
                        A.Normalize(
                            mean = [0.0, 0.0, 0.0],
                            std = [1.0, 1.0, 1.0],
                            max_pixel_value = 255.0
                            )
                        # Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                        ])

    dataset = CatDogMiniDataset(image_dir,train_transform)
    img, target_label = dataset[2]
    print(img)
    print(img.shape)