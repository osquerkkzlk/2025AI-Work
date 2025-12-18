import gradio as gr
import torch
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

@ torch.no_grad
def AppTransform_Photo2src(img,desc):
    #准备工作
    if not img:
        return
    shape=img.size
    net = CycleGAN(3, 3, configue)
    if f"G_{desc}_Photo2Src.pth" in  os.listdir(".\Storage"):
        net.G_src2Photo, net.G_Photo2src = load(net,desc)
    else:
        print(f"未找到  G_{desc}_Photo2Src.pth")
        return
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
    net=net.G_Photo2src
    net.eval()
    img=net(img)
    img=img.cpu().detach().clone()
    img=transforms.ToPILImage()(reverse_img(img[0])).convert("RGB")
    print("转换完成")
    return img.resize(shape)


@torch.no_grad
def AppTransform_src2Photo(img,desc):
    #准备工作
    shape=img.size
    net = CycleGAN(3, 3, configue)
    if f"G_{desc}_Src2Photo.pth" in  os.listdir(".\Storage"):
        net.G_src2Photo, net.G_Photo2src = load(net,desc)
    else:
        print(f"未找到  G_{desc}_src2Photo.pth")
        return
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
    net=net.G_src2Photo
    net.eval()
    img=net(img)
    img=img.cpu().detach().clone()
    img=transforms.ToPILImage()(reverse_img(img[0])).convert("RGB")
    print("转换完成")
    return img.resize(shape)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CycleGan --- src and your photo
        """
    )
    with gr.Sidebar(position="left"):
        gr.Markdown("# 配置选项")
        gr.Markdown("## 选择风格🙂‍↕️")
        gr.Markdown("""
                    ⚠️⚠️⚠️ 下方提供了示例，注意：风格图片还原为原始照片时必须选择其默认的风格，否则效果大打折扣。
                    而给照片施加某种风格则可以选择任意风格。


                    由于Gradio V5.50.0 Examples组件分页存在bug,所以第二页现实的风格和实际的风格存在差异，
                    但是左侧的按钮还是会正常随着照片跳转的。""")
        Style=gr.Radio(choices=["Monet","Ghibli","Shinkai"],
                       label="Style",
                       value="Monet")

    with gr.Tab("Source--->Style"):
        gr.Markdown("## 将图像转换为具有某种风格的图像，快来试试吧👀 ")
        with gr.Row(equal_height=True):
            A_image_input=gr.Image(label="原始图像",type="pil",height=256)
            A_image_output=gr.Image(label="某种风格的图像",interactive=False,height=256)
        A_convert_button=gr.Button("Start to convert 😲",interactive=True)
        gr.Examples(examples=ExampleShow()[0],
                    inputs=[A_image_input,Style],
                    examples_per_page=6
                    )
    with gr.Tab("Style--->Source"):
        gr.Markdown("## 将某种风格地图像还原为原始图像，快来试试吧👀")
        with gr.Row(equal_height=True):
            B_image_input=gr.Image(label="风格图像",type="pil",height=256)
            B_image_output=gr.Image(label="转换后的图像",interactive=False,height=256)
        B_convert_button=gr.Button("Start to convert 😲",interactive=True)
        gr.Examples(examples=ExampleShow()[1],
                    inputs=[B_image_input,Style],
                    examples_per_page=6
                    )
    A_convert_button.click(AppTransform_Photo2src,inputs=[A_image_input,Style],outputs=[A_image_output],
                           )
    B_convert_button.click(AppTransform_src2Photo,inputs=[B_image_input,Style],outputs=[B_image_output],
                           )

demo.launch()

