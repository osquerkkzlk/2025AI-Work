import os
import torch
from build_model import CycleGAN
from utils import show_img,show_chart,Transformer_image,save,load,photo_iter
import random
import numpy as np
from build_model import monet_iter

configue = {
    "batch_size": 4,
    "seed": 42,
    "epochs": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "start_lr": 2e-4,
    "temp": 10,
    "idt_coef": 0.5,
    "decay_epoch": 0,
    "training": True,
    "save": False
}

if __name__ == '__main__':

    # 设置种子
    def seed_(seed):
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        # 采用混合精度训练
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True


    seed_(configue["seed"])
    os.makedirs(".\Image", exist_ok=True)
    os.makedirs(".\Storage",exist_ok=True)

    net=CycleGAN(3,3,configue)
    # 两个权重同时保存，所以只需要判断一个即可
    if "G_Photo2Monet.pth" in  os.listdir(".\Storage"):
        net.G_Monet2Photo,net.G_Photo2Monet=load(net)

    if configue['training']:
        net.train(monet_iter)
        transformed_imgs, true_imgs=Transformer_image(net, photo_iter, configue)
        show_img(transformed_imgs, true_imgs)
        show_chart(net)
    if configue["save"]:
        save(net)
