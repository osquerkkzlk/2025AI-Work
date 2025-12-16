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
        self.upsample = nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0
        )
        self.norm = nn.InstanceNorm2d(out_ch)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.GELU()

    def forward(self, X):
        return self.activation(self.dropout(self.norm(self.upsample(X))))



class Conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=2,\
                            use_leaky=True,use_init_norm=True,use_pad=True):
        super().__init__()
        if use_pad:
            self.conv_layer=nn.Conv2d(in_ch,out_ch,kernel_size,stride,1,bias=True)
        else:
            self.conv_layer=nn.Conv2d(in_ch,out_ch,kernel_size,stride,0,bias=True)

        if use_leaky:
            self.activation=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        else:
            self.activation=nn.GELU()

        if use_init_norm:
            self.norm=nn.InstanceNorm2d(out_ch)
        else:
            self.norm=nn.BatchNorm2d(out_ch)
    def forward(self,X):
        return self.activation(self.norm(self.conv_layer(X)))


class Res_Block(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.pad=nn.ReflectionPad2d(1)
        self.conv_block=Conv_block(ch,ch,3,1,False,True,False)
        self.conv=nn.Conv2d(ch,ch,3,1,0,bias=True)
        self.dropout=nn.Dropout(0.5)
        self.norm=nn.InstanceNorm2d(ch)
    def forward(self,X):
        X1=self.dropout(self.conv_block(self.pad(X)))
        X2=self.norm(self.conv(self.pad(X1)))
        return X2+X

# 基本模型
class Generator(nn.Module):
    def __init__(self,in_ch,out_ch,ResBlock_num=6):
        super().__init__()
        self.model=nn.Sequential(nn.ReflectionPad2d(3),
                           Conv_block(in_ch,64,7,1,False,True,False),
                           Conv_block(64,128,3,2,False,True,True),
                           Conv_block(128,256,3,2,False,True,True),
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
    def __init__(self,in_ch,num_layers=4):
        super().__init__()
        model=nn.Sequential(
            nn.Conv2d(in_ch,64,4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
        )
        for i in range(1,num_layers):
            in_ch=64*2**(i-1)           # ** 的优先级更高一点
            out_ch=in_ch*2
            if i ==num_layers-1:
                model.append(Conv_block(in_ch,out_ch,4,1,True,True,True))
            else :
                model.append(Conv_block(in_ch,out_ch,4,2,True,True,True))
        model.append(nn.Conv2d(out_ch,1,4,stride=1,padding=1))
        self.model=model
    def forward(self,X):
        return self.model(X)


def init_weight(m):
    class_name = m.__class__.__name__
    # hasattr检查该层是否有该属性，但并不表示该属性被激活
    if hasattr(m,"weight") and ("Conv" in class_name or "Linear" in class_name ):
        nn.init.normal_(m.weight.data, 0,0.02)
        if hasattr(m,"bias")  and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif "BatchNorm2d" in class_name:
        nn.init.normal_(m.weight, 1,std=0.02)
        nn.init.zeros_(m.bias)

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
        self.G_Style2Photo = Generator(in_ch, out_ch)
        self.G_Photo2Style = Generator(in_ch, out_ch)
        self.D_Style = Discriminator(in_ch)
        self.D_Photo = Discriminator(in_ch)

        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.G_adam = torch.optim.Adam(
            itertools.chain(self.G_Style2Photo.parameters(), self.G_Photo2Style.parameters()),
            lr=self.start_lr, betas=(0.5, 0.999))
        self.D_adam = torch.optim.Adam(itertools.chain(self.D_Style.parameters(), self.D_Photo.parameters()),
                                        lr=self.start_lr, betas=(0.5, 0.999))

        self.sample_Style = sample_fake()
        self.sample_photo = sample_fake()

        G_lr = lr_sched(self.decay_epoch, self.epochs)
        D_lr = lr_sched(self.decay_epoch, self.epochs)
        self.G_sched = torch.optim.lr_scheduler.LambdaLR(self.G_adam, G_lr.step)
        self.D_sched = torch.optim.lr_scheduler.LambdaLR(self.D_adam, D_lr.step)
        self.record_metric = Recorder(2)

        self.init_model()

    def init_model(self):
        self.G_Style2Photo.apply(init_weight)
        self.G_Photo2Style.apply(init_weight)
        self.D_Style.apply(init_weight)
        self.D_Photo.apply(init_weight)

        self.G_Style2Photo = self.G_Style2Photo.to(self.device)
        self.G_Photo2Style = self.G_Photo2Style.to(self.device)
        self.D_Style = self.D_Style.to(self.device)
        self.D_Photo = self.D_Photo.to(self.device)

    def train(self, data_iter):
        epochs = self.configue["epochs"]
        device = self.configue["device"]

        record_metric = self.record_metric
        scaler = amp.GradScaler("cuda")
        for epoch in range(epochs):
            add_metric = Accumulator(3)
            pbar = tqdm.tqdm(data_iter, total=len(data_iter), leave=False, desc=f"{epoch + 1}/{epochs}")
            for i, (Style_img, photo_img) in enumerate(pbar):
                # ============================生成器============================
                Frozen([self.D_Style, self.D_Photo], False)
                Style_img, photo_img = Style_img.to(device, non_blocking=True), photo_img.to(device, non_blocking=True)
                self.G_adam.zero_grad()
                # 混合精度的前向传播
                with amp.autocast("cuda",enabled=True):
                    fake_photo = self.G_Style2Photo(Style_img)
                    fake_Style = self.G_Photo2Style(photo_img)
                    cycle_Style = self.G_Photo2Style(fake_photo)
                    cycle_photo = self.G_Style2Photo(fake_Style)
                    id_Style = self.G_Photo2Style(Style_img)
                    id_photo = self.G_Style2Photo(photo_img)

                    # 身份损失---不能改变本来就是目标身份的图片
                    id_Style_loss = self.l1loss(id_Style, Style_img) * self.temp * self.idt_coef
                    id_photo_loss = self.l1loss(id_photo, photo_img) * self.temp * self.idt_coef
                    # 循环损失---确保内容主体不改变，否则生成器只会骗判别器而全然不管内容主体，因此大权重
                    cycle_Style_loss = self.l1loss(cycle_Style, Style_img) * self.temp
                    cycle_photo_loss = self.l1loss(cycle_photo, photo_img) * self.temp
                    # 对抗损失---判别器能否正确判断真假，这是我们严格要求的，所以加上 mseloss，即不允许容忍极端值
                    Style_disc = self.D_Style(fake_Style)
                    photo_disc = self.D_Photo(fake_photo)
                    real = torch.ones(Style_disc.size()).to(device)
                    Style_adv_loss = self.mseloss(Style_disc, real)
                    photo_adv_loss = self.mseloss(photo_disc, real)
                    # 总损失
                    G_total_loss = id_Style_loss + id_photo_loss + cycle_Style_loss + \
                                   cycle_photo_loss + Style_adv_loss + photo_adv_loss
                scaler.scale(G_total_loss).backward()
                # 因为上述损失与生成器目标一致，所以我们希望优化器同时兼顾这些损失
                scaler.unscale_(self.G_adam)
                scaler.step(self.G_adam)
                scaler.update()

                # ============================判别器============================
                Frozen([self.D_Style, self.D_Photo], True)
                self.D_adam.zero_grad()

                with amp.autocast("cuda",enabled=True):
                    fake_Style = self.sample_Style(fake_Style.detach().clone()).to(device)
                    fake_photo = self.sample_photo(fake_photo.detach().clone()).to(device)

                    Style_real_disc = self.D_Style(Style_img)
                    Style_fake_disc = self.D_Style(fake_Style)
                    photo_real_disc = self.D_Photo(photo_img)
                    photo_fake_disc = self.D_Photo(fake_photo)

                    # 通用的判别值
                    real_dsic = torch.ones_like(Style_real_disc, device=device)
                    fake_disc = torch.zeros_like(Style_fake_disc, device=device)

                    # 真的判别为真，假的判别为假
                    Style_real_loss = self.mseloss(Style_real_disc, real_dsic)
                    Style_fake_loss = self.mseloss(Style_fake_disc, fake_disc)
                    photo_real_loss = self.mseloss(photo_real_disc, real_dsic)
                    photo_fake_loss = self.mseloss(photo_fake_disc, fake_disc)
                    # 总损失
                    Style_disc_loss = (Style_real_loss + Style_fake_loss) / 2
                    photo_disc_loss = (photo_real_loss + photo_fake_loss) / 2
                    D_total_loss = Style_disc_loss + photo_disc_loss
                scaler.scale(D_total_loss).backward()
                scaler.unscale_(self.D_adam)
                scaler.step(self.D_adam)
                scaler.update()

                add_metric.add(G_total_loss.item(), D_total_loss.item(), len(Style_img))
                # 进度条更新
                pbar.set_postfix(G_loss=f"{G_total_loss.item():.4f}", D_loss=f"{D_total_loss.item():.4f}")
            record_metric.add(add_metric[0] / add_metric[-1], add_metric[1] / add_metric[-1])

            self.G_sched.step()
            self.D_sched.step()
            print(f"<{epoch+1}/{epochs}>  G_loss--->{record_metric[0][-1]}   D_loss--->{record_metric[1][-1]}")
            if self.configue["show_in_training"] and (epoch + 1) % self.configue["show_per_epochs"] == 0:
                self.visualize_result()

    def visualize_result(self):
        self.G_Photo2Style.eval()
        # 创建一个临时的小 loader 用于展示
        img = next(iter(self.show_iter))[1].to(self.device)

        with torch.no_grad():
            fake = self.G_Photo2Style(img)

        # 转回 CPU 显式
        fake_img = reverse_img(fake.cpu()[0]).permute(1, 2, 0).numpy()
        real_img = reverse_img(img.cpu()[0]).permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Photo")
        plt.imshow(real_img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Generated Style")
        plt.imshow(fake_img)
        plt.axis("off")
        plt.show()
        plt.close()
        self.G_Photo2Style.train()


