## 项目简介
我们通过CycleGan技术实现了画风风格和真实图片的互相转换
，并通过Gradio托管到[huggingface space](https://huggingface.co/spaces/linyuZheng/StyleTransfer_BY_CycleGan)上，实现了
在线风格转换。同时，我们提供了莫奈,吉卜力,新海诚的风格供使用者选择。


## 项目框架

```markdown
CycleGan/
├── build_model.py      # CycleGAN 模型构建（Generator + Discriminator）
├── utils.py            # 工具函数文件（数据预处理、可视化等）
├── data.py             # 数据加载相关的函数文件
├── app.py              # Web 演示界面（基于 Gradio，可独立运行）
├── main.py             # 训练主脚本（训练入口）
├── README.md           # 项目说明文档
├── requirements.txt    # Python 依赖环境说明
│
├── Storage/      # 模型权重以及损失曲线文件夹
│   ├──G_Ghibli_Photo2Src.pth
│   ├──G_Ghibli_Src2Photo.pth
│   ├──G_Monet_Photo2Src.pth             
│   ├──G_Monet_Src2Photo.pth
│   ├──Ghibli_loss_curve.png
│   └──Monet_loss_curve.png  
│
├── Image/         # 图片保存文件,存放前端示例图片
│   ├──Ghibli
│   ├──Monet
│   └──Shinkai      
│  
└── data/           # 数据集下载位置以及注意事项
    ├──Ghibli
    ├──Monet
    ├──Shinkai       
    └──Address.txt  
 
```

## 说用说明
1. 可以选择运行app.py，可在网页上在线转换。
2. 可以在我们提供的huggingface链接上随时转换。
3. Gradio V5.50.0 Examples组件存在bug，示例图片的Style未必为真，但是不影响使用。

## 成果预览
### V1

可以明显看出出现了棋盘效应,过渡不自然---原因：上采样中使用了转置卷积，由于图像尺寸不能被步长整除，出现的错位现象
### V2
对项目进行了整体升级，优化了代码逻辑以尽可能消除棋盘效应的影响，优化了前端界面显示，新增了吉卜力风格和新海诚风格。


| 原始图片 (Original)                     | 转换后图片 (CycleGAN Generated)            |
|-------------------------------------|---------------------------------------|
| ![original](.\Image\Original_1.png) | ![generated](.\Image\Generated_1.png) || <img src="results/fake_B/001.jpg" width="400"/> |
| ![original](.\Image\Original_2.png) | ![generated](.\Image\Generated_2.png) || <img src="results/fake_B/001.jpg" width="400"/> |