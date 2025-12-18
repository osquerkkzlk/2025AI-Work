import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms
import random

# 全局配置
configue = {
    "batch_size": 1,
    "seed": 42,
    "epochs": 100,   # 根据数据量酌情更改 ，供参考-->   Monet epochs==100,len(src)==300
                                                # Shinkai & Ghibli epochs=30,len(src)=2500
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "start_lr": 2e-4,
    "temp": 10,
    "idt_coef": 0.5,
    "decay_epoch": 60,

    "save":True ,            # 是否保存生成器权重，默认不保存，使用已经训练好的权重
    "training": True,         # 控制是否训练
    "show_per_epochs":5,      # 每过几个epoch就显示一次，同参数 "show_in_training"相配合
    "show_in_training":True  # 控制训练过程中是否显示 Src转换图,但是pycahrm容易堵塞进程，即必须主动关闭图像窗口才能继续训练
}

# 设置种子
def seed_everything(seed):
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # 采用混合精度训练
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def reverse_img(img,mean=[0.5]*3,std=[0.5]*3):
    for img_channel,mean_channel,std_channel in zip(img,mean,std):
        img_channel.mul_(std_channel).add_(mean_channel)
    return img.clamp(0,1)

# 预测和显示函数
def show_img(transformed_imgs, true_imgs):
    if len(transformed_imgs) == 0: return
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(true_imgs[-1])
    plt.axis("off")
    plt.title("Original (Photo)")

    plt.subplot(122)
    plt.imshow(transformed_imgs[-1])
    plt.axis("off")
    plt.title("Generated (Src-esque)")

    plt.show()
    plt.close()

def show_chart(net,desc):
    series=np.arange(len(net.record_metric[0]))
    plt.plot(series,net.record_metric[0],"r-",label="G_loss")
    plt.plot(series,net.record_metric[1],"b-",label="D_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(f"{desc}_loss_curve")
    plt.savefig(f"./Storage/{desc}_loss_curve")
    plt.show()

@torch.no_grad
def Transformer_image(net, data_iter, configue=configue, num=1):
    net.G_src2Photo.eval()
    net.G_Photo2src.eval()
    net.D_src.eval()
    net.D_Photo.eval()
    true_imgs, transformed_imgs = [], []
    for i, img in enumerate(data_iter):
        if i == num:
            break
        device = configue["device"]
        img = img.to(device)
        transformed_img = net.G_Photo2src(img)
        transformed_img = reverse_img(transformed_img.cpu()[0])
        transformed_img = transforms.ToPILImage()(transformed_img).convert("RGB")
        transformed_imgs.append(transformed_img)

        img = reverse_img(img.cpu())
        img = transforms.ToPILImage()(img[0]).convert("RGB")
        true_imgs.append(img)

        transformed_img.save(f"Image/image_{i}.png")
    return transformed_imgs, true_imgs

def load(net,desc):
    net.G_src2Photo.load_state_dict(torch.load(f"./Storage/G_{desc}_Src2Photo.pth"))
    net.G_Photo2src.load_state_dict(torch.load(f"./Storage/G_{desc}_Photo2Src.pth"))

    return net.G_src2Photo,net.G_Photo2src

def save(G_Src2Photo,G_Photo2Src,desc):
    torch.save(G_Src2Photo.state_dict(),f"./Storage/G_{desc}_Src2Photo.pth")
    torch.save(G_Photo2Src.state_dict(), f"./Storage/G_{desc}_Photo2Src.pth")

class sample_fake(object):
    def __init__(self, max_imgs=50):
        self.max_imgs = max_imgs
        self.imgs = []

    def __call__(self, imgs):
        ret = []
        for img in imgs:
            if len(self.imgs) < self.max_imgs:
                self.imgs.append(img)
                ret.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.max_imgs)
                    ret.append(self.imgs[idx])
                    self.imgs[idx] = img
                else:
                    ret.append(img)
        return torch.stack(ret)

class lr_sched():
    def __init__(self, decay_epochs=100, total_epochs=200):
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs

    def step(self, epoch_num):
        if epoch_num <= self.decay_epochs:
            return 1.0
        else:
            fract = (epoch_num - self.decay_epochs)  / (self.total_epochs - self.decay_epochs)
            return max(0 , 1.0 - fract)

class Accumulator:
    def __init__(self,num):
        self.metric=[0]*num
    def __getitem__(self,idx):
        return self.metric[idx]
    def add(self,*args):
        if args:
            for i,arg in enumerate(args):
                self.metric[i]+=arg

class Recorder:
    def __init__(self,num):
        self.metric=[[] for _ in range(num)]
    def __getitem__(self,idx):
        return self.metric[idx]
    def __len__(self):
        return len(self.metric)
    def add(self,*args):
        if args:
            for i,arg in enumerate(args):
                self.metric[i].append(arg)

def show_configue():
    print("======当前配置======")
    for key,value in configue.items():
        print(f"{key:20} {value}")
    print("\n\n")

