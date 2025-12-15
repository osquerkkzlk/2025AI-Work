from torch.utils.data import Dataset
import random
import glob
from torchvision import transforms
from PIL import Image
import os
import zipfile


# 数据集搭建
def extract():
    # 解压缩
    dir_path = r".\data\data.zip"
    extract_path = r".\data"
    with zipfile.ZipFile(dir_path, "r") as f:
        f.extractall(extract_path)

    print("解压缩完成")
    monet_path = os.path.join(extract_path, "data", "monet_jpg")
    photo_path = os.path.join(extract_path, "data", "photo_jpg")
    return monet_path,photo_path

class PhotoDataset(Dataset):
    def __init__(self, photo_path,configue, size=(256, 256)):
        super().__init__()
        random.seed(None)
        self.photo_path = glob.glob(photo_path + "/*")
        random.shuffle(self.photo_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(286),
            transforms.RandomCrop(size),
            transforms.Resize(size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        random.seed(configue["seed"])

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, idx):
        photo_img = self.transforms(Image.open(self.photo_path[idx]))
        return photo_img

class MonetDataset(Dataset):
    def __init__(self,monet_path,photo_path,configue,size=(256,256)):
        super().__init__()
        random.seed(None)
        self.monet_path=glob.glob(monet_path+"\*")
        self.photo_path=glob.glob(photo_path+"\*")
        random.shuffle(self.monet_path)
        random.shuffle(self.photo_path)
        self.transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(286),
            transforms.RandomCrop(size),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        random.seed(configue["seed"])
    def __len__(self):
        return max(len(self.monet_path),len(self.photo_path))

    def __getitem__(self,idx):
        monet_img=self.transforms(Image.open(self.monet_path[idx % len(self.monet_path)]))
        photo_img=self.transforms(Image.open(self.photo_path[idx]))
        return monet_img,photo_img

