import torch
from torch import nn
from torch import amp
import itertools
import tqdm
from utils import sample_fake,Recorder,Accumulator,lr_sched,reverse_img
import matplotlib.pyplot as plt

# 网络架构块
class Upsample_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.activation = nn.GELU()

    def forward(self, X):
        return self.activation(self.norm(self.conv(self.upsample(X))))

class Conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,stride=2):
        super().__init__()
        self.conv_layer=nn.Conv2d(in_ch,out_ch,kernel_size,stride=stride,padding=padding)
        self.activation=nn.GELU()
        self.norm=nn.InstanceNorm2d(out_ch)
    def forward(self,X):
        return self.activation(self.norm(self.conv_layer(X)))

class Res_Block(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.pad=nn.ReflectionPad2d(1)
        self.conv=nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=0)
        self.norm=nn.InstanceNorm2d(ch)
        self.activation=nn.GELU()
    def forward(self,X):
        X1=self.activation(self.norm(self.conv(self.pad(X))))
        X2=self.norm(self.conv(self.pad(X1)))
        return X2+X

# 基本模型
class Generator(nn.Module):
    def __init__(self,in_ch,out_ch,ResBlock_num=9):
        super().__init__()
        self.model=nn.Sequential(nn.ReflectionPad2d(3),
                           Conv_block(in_ch,64,7,stride=1,padding=0),
                           Conv_block(64,128,3,stride=2,padding=1),
                           Conv_block(128,256,3,stride=2,padding=1),
                            # 抽取特征
                            *[Res_Block(256) for _ in range(ResBlock_num)],
                            # 上采样
                            Upsample_block(256,128),
                            Upsample_block(128,64),
                            #抽取全局特征
                            nn.ReflectionPad2d(3),
                            nn.Conv2d(64,out_ch,7,stride=1,padding=0),
                            nn.Tanh()
                           )
    def forward(self,X):
        return self.model(X)

# PatchGAN
class Discriminator(nn.Module):
    def __init__(self,in_ch,num_layers=3):
        super().__init__()
        model=nn.Sequential(
            nn.Conv2d(in_ch,64,4,stride=2,padding=1),
            nn.GELU(),
        )
        for i in range(1,num_layers+1):
            in_ch=64*(2**(i-1))
            out_ch=in_ch*2
            if i ==num_layers:
                model.append(Conv_block(in_ch,out_ch,4,2))
            else :
                model.append(Conv_block(in_ch,out_ch,4,1))
        model.append(nn.Conv2d(out_ch,1,4,stride=1,padding=1))
        self.model=model
    def forward(self,X):
        return self.model(X)

def init_weight(m):
    class_name = m.__class__.__name__
    if class_name in ["Conv2d"]:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if class_name == "Conv2d" and m.out_channels == 3:
        nn.init.normal_(m.weight, std=0.02)

def Frozen(nets,training=True):
    for net in nets:
        for param in net.parameters():
                param.requires_grad=training

#关键入口函数
class CycleGAN:
    def __init__(self, in_ch, out_ch, configue,show_iter=None):
        self.configue = configue
        self.epochs = configue["epochs"]
        self.start_lr = configue["start_lr"]
        self.temp = configue["temp"]
        self.decay_epoch = configue["decay_epoch"] if configue["decay_epoch"] else int(self.epochs / 2)
        self.idt_coef = configue["idt_coef"]
        self.device = configue["device"]
        self.show_iter=show_iter
        self.G_Monet2Photo = Generator(in_ch, out_ch)
        self.G_Photo2Monet = Generator(in_ch, out_ch)
        self.D_Monet = Discriminator(in_ch)
        self.D_Photo = Discriminator(in_ch)

        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.G_adam = torch.optim.AdamW(
            itertools.chain(self.G_Monet2Photo.parameters(), self.G_Photo2Monet.parameters()),
            lr=self.start_lr, betas=(0.5, 0.999))
        self.D_adam = torch.optim.AdamW(itertools.chain(self.D_Monet.parameters(), self.D_Photo.parameters()),
                                        lr=self.start_lr, betas=(0.5, 0.999))

        self.sample_monet = sample_fake()
        self.sample_photo = sample_fake()

        G_lr = lr_sched(self.decay_epoch, self.epochs)
        D_lr = lr_sched(self.decay_epoch, self.epochs)
        self.G_sched = torch.optim.lr_scheduler.LambdaLR(self.G_adam, G_lr.step)
        self.D_sched = torch.optim.lr_scheduler.LambdaLR(self.D_adam, D_lr.step)
        self.record_metric = Recorder(2)

        self.init()

    def init(self):
        self.G_Monet2Photo.apply(init_weight)
        self.G_Photo2Monet.apply(init_weight)
        self.D_Monet.apply(init_weight)
        self.D_Photo.apply(init_weight)

        self.G_Monet2Photo = self.G_Monet2Photo.to(self.device)
        self.G_Photo2Monet = self.G_Photo2Monet.to(self.device)
        self.D_Monet = self.D_Monet.to(self.device)
        self.D_Photo = self.D_Photo.to(self.device)

    def train(self, data_iter):
        epochs = self.configue["epochs"]
        device = self.configue["device"]

        record_metric = self.record_metric
        scaler = amp.GradScaler("cuda")
        for epoch in range(epochs):
            add_metric = Accumulator(3)
            pbar = tqdm.tqdm(data_iter, total=len(data_iter), leave=False, desc=f"{epoch + 1}/{epochs}")
            for i, (monet_img, photo_img) in enumerate(pbar):
                # ============================生成器============================
                Frozen([self.D_Monet, self.D_Photo], False)
                monet_img, photo_img = monet_img.to(device, non_blocking=True), photo_img.to(device, non_blocking=True)
                self.G_adam.zero_grad()
                # 混合精度的前向传播
                with amp.autocast("cuda",enabled=True):
                    fake_photo = self.G_Monet2Photo(monet_img)
                    fake_monet = self.G_Photo2Monet(photo_img)
                    cycle_monet = self.G_Photo2Monet(fake_photo)
                    cycle_photo = self.G_Monet2Photo(fake_monet)
                    id_monet = self.G_Photo2Monet(monet_img)
                    id_photo = self.G_Monet2Photo(photo_img)

                    # 身份损失---不能改变本来就是目标身份的图片
                    id_monet_loss = self.l1loss(id_monet, monet_img) * self.temp * self.idt_coef
                    id_photo_loss = self.l1loss(id_photo, photo_img) * self.temp * self.idt_coef
                    # 循环损失---确保内容主体不改变，否则生成器只会骗判别器而全然不管内容主体，因此大权重
                    cycle_monet_loss = self.l1loss(cycle_monet, monet_img) * self.temp
                    cycle_photo_loss = self.l1loss(cycle_photo, photo_img) * self.temp
                    # 对抗损失---判别器能否正确判断真假，这是我们严格要求的，所以加上 mseloss，即不允许容忍极端值
                    monet_disc = self.D_Monet(fake_monet)
                    photo_disc = self.D_Photo(fake_photo)
                    real = torch.ones(monet_disc.size()).to(device)
                    monet_adv_loss = self.mseloss(monet_disc, real)
                    photo_adv_loss = self.mseloss(photo_disc, real)
                    # 总损失
                    G_total_loss = id_monet_loss + id_photo_loss + cycle_monet_loss + \
                                   cycle_photo_loss + monet_adv_loss + photo_adv_loss
                scaler.scale(G_total_loss).backward()
                # 因为上述损失与生成器目标一致，所以我们希望优化器同时兼顾这些损失
                scaler.unscale_(self.G_adam)
                scaler.step(self.G_adam)
                scaler.update()

                # ============================判别器============================
                Frozen([self.D_Monet, self.D_Photo], True)
                self.D_adam.zero_grad()

                with amp.autocast("cuda",enabled=True):
                    fake_monet = self.sample_monet(fake_monet.detach().clone()).to(device)
                    fake_photo = self.sample_photo(fake_photo.detach().clone()).to(device)

                    monet_real_disc = self.D_Monet(monet_img)
                    monet_fake_disc = self.D_Monet(fake_monet)
                    photo_real_disc = self.D_Photo(photo_img)
                    photo_fake_disc = self.D_Photo(fake_photo)

                    # 通用的判别值
                    real_dsic = torch.ones_like(monet_real_disc, device=device)
                    fake_disc = torch.zeros_like(monet_fake_disc, device=device)

                    # 真的判别为真，假的判别为假
                    monet_real_loss = self.mseloss(monet_real_disc, real_dsic)
                    monet_fake_loss = self.mseloss(monet_fake_disc, fake_disc)
                    photo_real_loss = self.mseloss(photo_real_disc, real_dsic)
                    photo_fake_loss = self.mseloss(photo_fake_disc, fake_disc)
                    # 总损失
                    monet_disc_loss = (monet_real_loss + monet_fake_loss) / 2
                    photo_disc_loss = (photo_real_loss + photo_fake_loss) / 2
                    D_total_loss = monet_disc_loss + photo_disc_loss
                scaler.scale(D_total_loss).backward()
                scaler.unscale_(self.D_adam)
                scaler.step(self.D_adam)
                scaler.update()

                add_metric.add(G_total_loss.item(), D_total_loss.item(), len(monet_img))
                # 进度条更新
                pbar.set_postfix(G_loss=f"{G_total_loss.item():.4f}", D_loss=f"{D_total_loss.item():.4f}")
            record_metric.add(add_metric[0] / add_metric[-1], add_metric[1] / add_metric[-1])

            self.G_sched.step()
            self.D_sched.step()

            if self.configue["show_in_training"] and (epoch + 1) % self.configue["show_per_epochs"] == 0:
                self.visualize_result()

    def visualize_result(self):
        self.G_Photo2Monet.eval()
        # 创建一个临时的小 loader 用于展示
        img = next(iter(self.show_iter)).to(self.device)

        with torch.no_grad():
            fake = self.G_Photo2Monet(img)

        # 转回 CPU 显式
        fake_img = reverse_img(fake.cpu()[0]).permute(1, 2, 0).numpy()
        real_img = reverse_img(img.cpu()[0]).permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Generated Monet")
        plt.imshow(fake_img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Original Photo")
        plt.imshow(real_img)
        plt.axis("off")
        plt.show()
        plt.close()

        self.G_Photo2Monet.train()


