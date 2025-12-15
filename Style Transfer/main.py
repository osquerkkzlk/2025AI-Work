import torch
from train import train
from utils import save,load_image
import os
from glob import glob

configue = {
    "epochs": 5000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "steps": 3000,
    "epoch_step": 1000  # 每多少步保存一次中间结果
}

if __name__ == '__main__':
    print("training...")
    os.makedirs("./Storage",exist_ok=True)

    content_image=load_image(glob("./Image/content*")[0]).to(configue["device"])
    style_img=load_image(glob("./Image/style*")[0]).to(configue["device"])


    X,imgs=train(configue,content_image,style_img)
    if input("是否保存最终图像,1为保存，否则不保存")=="1":
        save(imgs,configue)