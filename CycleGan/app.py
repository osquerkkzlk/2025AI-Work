import gradio as gr
from build_model import CycleGAN
from utils import load,configue
import os
import glob
from torchvision import transforms
from utils import reverse_img
from pathlib import Path

def ExampleShow():
    return [[path,Path(path).parts[-2]]for path in glob.glob("./Image/*/Photo*",recursive=True)], \
        [[path, Path(path).parts[-2]] for path in glob.glob("./Image/*/Style*", recursive=True)]
print(ExampleShow())
def AppTransform_Photo2Style(img,desc):
    #准备工作
    shape=img.size
    net = CycleGAN(3, 3, configue)
    if f"G_{desc}_Photo2Style.pth" in  os.listdir(".\Storage"):
        net.G_Style2Photo, net.G_Photo2Style = load(net,desc)
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
    net=net.G_Photo2Style
    net.eval()
    img=net(img)
    img=img.cpu().detach().clone()
    img=transforms.ToPILImage()(reverse_img(img[0])).convert("RGB")
    print("转换完成")
    return img.resize(shape)

def AppTransform_Style2Photo(img,desc):
    #准备工作
    shape=img.size
    net = CycleGAN(3, 3, configue)
    if f"G_{desc}_Style2Photo.pth" in  os.listdir(".\Storage"):
        net.G_Style2Photo, net.G_Photo2Style = load(net,desc)
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
    net=net.G_Style2Photo
    net.eval()
    img=net(img)
    img=img.cpu().detach().clone()
    img=transforms.ToPILImage()(reverse_img(img[0])).convert("RGB")
    print("转换完成")
    return img.resize(shape)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CycleGan --- Style and your photo
        """
    )
    with gr.Sidebar(position="left"):
        gr.Markdown("# 配置选项")
        gr.Markdown("## 选择风格🙂‍↕️")
        gr.Markdown("⚠️⚠️⚠️ 下方提供了示例，注意：风格图片还原为原始照片时必须选择其默认的风格，否则效果大打折扣")
        style=gr.Radio(choices=["Monet","Ghibli","Shinkai"],
                       label="Style",
                       value="Monet")
    with gr.Tab("Source--->style"):
        gr.Markdown("## 将图像转换为具有某种风格的图像，快来试试吧👀 ")
        with gr.Row(equal_height=True):
            A_image_input=gr.Image(label="原始图像",height=256)
            A_image_output=gr.Image(label="转换后具有某种风格的图像",interactive=False,height=256)
        A_convert_button=gr.Button("Start to convert 😲",interactive=True)
        gr.Examples(examples=ExampleShow()[0],
                    inputs=[A_image_input,style],
                    examples_per_page=6
                    )
    with gr.Tab("Style--->Source"):
        gr.Markdown("## 将某种风格地图像还原为原始图像，快来试试吧👀")
        with gr.Row(equal_height=True):
            B_image_input=gr.Image(label="风格图像",height=256)
            B_image_output=gr.Image(label="转换后的图像",interactive=False,height=256)
        B_convert_button=gr.Button("Start to convert 😲",interactive=True)
        gr.Examples(examples=ExampleShow()[1],
                    inputs=[B_image_input,style],
                    examples_per_page=6
                    )

    A_convert_button.click(AppTransform_Photo2Style,inputs=[A_image_input,style],outputs=[A_image_output],
                           )
    B_convert_button.click(AppTransform_Style2Photo,inputs=[B_image_input,style],outputs=[B_image_output],
                           )

demo.launch()

