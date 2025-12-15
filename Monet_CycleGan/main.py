import os
from build_model import CycleGAN
from utils import show_img,show_chart,Transformer_image,save,configue,seed_everything
from data import extract,PhotoDataset,MonetDataset
from torch.utils.data import DataLoader



if __name__ == '__main__':
    configue=configue
    seed_everything(configue["seed"])

    os.makedirs(".\Image", exist_ok=True)
    os.makedirs(".\Storage",exist_ok=True)

    # 数据整理
    monet_path, photo_path = extract()
    photo_iter = DataLoader(PhotoDataset(photo_path,configue), batch_size=1,
                            shuffle=True, pin_memory=True)
    monet_iter = DataLoader(MonetDataset(monet_path, photo_path,configue),
                            batch_size=configue["batch_size"],
                            shuffle=True,pin_memory=True, )

    # 模型构建与训练入口
    net=CycleGAN(3,3,configue,photo_iter)
    if configue['training']:
        net.train(monet_iter)
        transformed_imgs, true_imgs=Transformer_image(net, photo_iter, configue)
        show_img(transformed_imgs, true_imgs)
        show_chart(net)
    if configue["save"]:
        save(net)


