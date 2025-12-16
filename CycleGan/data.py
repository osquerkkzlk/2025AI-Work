from torch.utils.data import Dataset
import random
import glob
from torchvision import transforms
from PIL import Image
import os
import zipfile


# 数据集搭建
def extract(dir_path,extract_path):
    with zipfile.ZipFile(dir_path, "r") as f:
        f.extractall(extract_path)
    print("解压缩完成")

class ImageDataset(Dataset):
    def __init__(self,Style_path,photo_path,size=(256,256)):
        super().__init__()
        self.Style_path=glob.glob(Style_path+"\*")
        self.photo_path=glob.glob(photo_path+"\*")
        self.transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(286),
            transforms.RandomCrop(size),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    def __len__(self):
        return min(len(self.Style_path),len(self.photo_path))

    def __getitem__(self,idx):
        photo_idx=random.randint(0,len(self.photo_path))
        Style_img=self.transforms(Image.open(self.Style_path[idx]).convert("RGB"))
        photo_img=self.transforms(Image.open(self.photo_path[photo_idx]).convert("RGB"))
        return Style_img,photo_img

def choose_task():
    dt={1:"Monet",2:"Ghibli",3:"Shinkai"}
    os.makedirs("./data",exist_ok=True)

    task=input("whitch training task do you want to choose\n"
               "♦️ 1--------------------Monet\n"
               "♦️ 2--------------------Ghibli\n"
               "♦️ 3--------------------Shinkai\n")
    match task:
        case "q":
            return
        case "1":
            if glob.glob("./data/Monet.zip"):
                os.makedirs("./data/Monet",exist_ok=True)
                extract("./data/Monet.zip","./data")
                return os.path.join("./data/Monet","monet"),os.path.join("./data/Monet","photo"),"Monet"
            else:
                print("can't find this file ")
                return
        case "2":
            if glob.glob("./data/Ghibli.zip"):
                os.makedirs("./data/Ghibli",exist_ok=True)
                extract("./data/Ghibli.zip","./data")
                return os.path.join("./data/Ghibli", "ghibli"), os.path.join("./data/Ghibli", "photo"),"Ghibli"
            else:
                print("can't find this file ")
                return
        case "3":
            if glob.glob("./data/Shinkai.zip"):
                os.makedirs("./data/Shinkai",exist_ok=True)
                extract("./data/Shinkai.zip","./data")
                return os.path.join("./data/Shinkai", "shinkai"), os.path.join("./data/Shinkai", "photo"),"Shinkai"
            else:
                print("can't find this file ")
                return
        case _:
            print("=====指令错误，重新输入，按 q 直接退出=====")
            choose_task()
            return
