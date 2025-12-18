import os
from build_model import CycleGAN
from utils import show_img,show_chart,Transformer_image,save,configue,seed_everything,show_configue
from data import choose_task,ImageDataset
from torch.utils.data import DataLoader



if __name__ == '__main__':
    configue=configue
    seed_everything(configue["seed"])
    show_configue()

    os.makedirs(".\Image", exist_ok=True)
    os.makedirs(".\Storage",exist_ok=True)

    # 数据整理
    Style_path, photo_path ,desc= choose_task()
    image_iter = DataLoader(ImageDataset(Style_path,photo_path), batch_size=configue["batch_size"],
                            shuffle=True, pin_memory=True)

    # 模型构建与训练入口
    net=CycleGAN(3,3,configue,image_iter)
    if configue['training']:
        net.train(image_iter)
        # transformed_imgs, true_imgs=Transformer_image(net, image_iter, configue)
        # show_img(transformed_imgs, true_imgs)
        show_chart(net,desc)
    if configue["save"]:
        save(net,desc)


