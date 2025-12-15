import gradio as gr
from build_model import CycleGAN
from utils import load,configue
import os
from torchvision import transforms
from utils import reverse_img

def AppTransform_Photo2Monet(img):
    net = CycleGAN(3, 3, configue)
    if "G_Monet2Photo.pth" in os.listdir(".\Storage"):
        net.G_Monet2Photo, net.G_Photo2Monet = load(net)
    size = (256, 256)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img=transform(img)
    device = configue["device"]
    img = img.to(device)
    img=img.unsqueeze(0)

    #转换以及反转换
    net=net.G_Photo2Monet
    net.eval()
    img=net(img)
    img=img.cpu().detach().clone()
    img=transforms.ToPILImage()(reverse_img(img[0])).convert("RGB")
    print("转换完成")
    return img

def AppTransform_Monet2Photo(img):
    #准备工作
    net = CycleGAN(3, 3, configue,)
    if "G_Photo2Monet.pth" in  os.listdir(".\Storage"):
        net.G_Monet2Photo, net.G_Photo2Monet = load(net)
    size = (256, 256)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img=transform(img)
    device = configue["device"]
    img = img.to(device)
    img=img.unsqueeze(0)

    #转换以及反转换
    net=net.G_Monet2Photo
    net.eval()
    img=net(img)
    img=img.cpu().detach().clone()
    img=transforms.ToPILImage()(reverse_img(img[0])).convert("RGB")
    print("转换完成")
    return img

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CycleGan --- Monet and your photo
        """
    )
    with gr.Tab("Monet--->Photo"):
        with gr.Row(equal_height=True):
            A_image_input=gr.Image(label="莫奈风格的图像")
            A_image_output=gr.Image(label="转换后的还原图像",interactive=False)
        A_convert_button=gr.Button("Start to convert",interactive=True)
    with gr.Tab("Photo--->Monet style"):
        with gr.Row(equal_height=True):
            B_image_input=gr.Image(label="原始图像")
            B_image_output=gr.Image(label="转换后具有莫奈风格的图像",interactive=False)
        B_convert_button=gr.Button("Start to convert",interactive=True)
    B_convert_button.click(AppTransform_Photo2Monet,inputs=[B_image_input],outputs=[B_image_output],
                           )
    A_convert_button.click(AppTransform_Monet2Photo,inputs=[A_image_input],outputs=[A_image_output],
                           )
demo.launch()

