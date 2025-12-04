## 项目简介
我们通过CycleGan技术实现了画风风格和真实图片的互相转换
，并通过Gradio托管到[huaggingface space]()上，实现了
在线风格转换。同时，我们提供了莫奈,,,的风格供使用者选择。

我们提供了两种版本，即py版本（偏向工程）和jupyter版本（偏向教学），
使用者可以自行选择。

## 项目框架

```markdown
Money_CycleGan/py
├── app.py              # Web 演示界面（基于 Gradio，可独立运行）
├── build_model.py      # CycleGAN 模型构建（Generator + Discriminator）
├── data.zip            # 数据集压缩包（含莫奈风格图片和真实图片）
├── main.py             # 训练主脚本（训练入口）
├── README.md           # 项目说明文档
├── requirements.txt    # Python 依赖环境说明
├── Storage             # 模型权重保存文件夹
├── Image               # 图片保存文件
└── utils.py            # 工具函数（数据加载、预处理、可视化等）
```

## 说用说明
1. 直接运行main.py文件可以-开始训练，但可以选择是否导入现有权重，根据目的决定是否注释下方代码
    ```python
    if os.listdir(".\Storage"):
        net.G_Monet2Photo,net.G_Photo2Monet=load(net)
    ```
2. 可以选择运行app.py，可在网页上在线转换。
3. 可以在我们提供的huggingface链接上随时转换。

## 成果预览
V1：可以明显看出出现了棋盘效应,过渡不自然---原因：上采样中使用了转置卷积，由于图像尺寸不能被步长整除，出现的错位现象


| 原始图片 (Original)                                | 转换后图片 (CycleGAN Generated)                      |
|------------------------------------------------|-------------------------------------------------|
| ![original](.\Image\Original.png)              | ![generated](.\Image\Generated.png)             || <img src="results/fake_B/001.jpg" width="400"/> |