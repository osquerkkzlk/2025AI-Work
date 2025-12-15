import gradio as gr
import torch
from utils import load_image, Recorder, convert_image
from train import get, get_features, criterion
import tqdm

SIMPLE_PROGRESS_HTML_START = """<div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px;">"""
SIMPLE_PROGRESS_HTML_END = """</div>"""

configue = {
    "epochs": 5000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "steps": 3000,
    "epoch_step": 1000  # 这个参数在实时更新中不再重要
}
example_data = [
        ["Image/content1.jpg", "Image/style1.jpg"],
        ["Image/content2.jpg", "Image/style2.jpg"],
    ]




def transfer(content_path, style_path):
    device = configue["device"]
    epochs=configue["epochs"]


    content_image = load_image(content_path).to(device)
    style_image = load_image(style_path).to(device)
    content_X, style_grams, target, net = get(content_image, style_image, device)

    optim = torch.optim.AdamW([target], lr=0.003)
    loss_recorder = Recorder(4)

    pbar = tqdm.tqdm(total=epochs, desc="Training...")

    for epoch in range(epochs):
        target_features = get_features(target, net)
        optim.zero_grad()
        c_loss, s_loss, t_loss, total = criterion(target, content_X["conv4_1"], target_features["conv4_1"], \
                                                  style_grams, target_features)
        total.backward()
        optim.step()

        loss_recorder.add(c_loss.item(), s_loss.item(), t_loss.item(), total.item())
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss_recorder[3][-1]:.4f}")

        percent = (epoch + 1) / epochs
        progress_width = int(percent * 100)

        html_content = f"""
        {SIMPLE_PROGRESS_HTML_START}
            <div style="width: {progress_width}%; height: 100%; background-color: #4CAF50; border-radius: 10px; text-align: right; color: white;">
                {progress_width}%
            </div>
        {SIMPLE_PROGRESS_HTML_END}
        <p style="text-align:center; color: #555; margin-top:5px;">Epoch {epoch + 1}/{epochs} | Loss: {loss_recorder[3][-1]:.4f}</p>
        """

        current_img = convert_image(target)
        yield html_content, current_img


# --- Gradio 界面保持不变 ---
with gr.Blocks(title="你想要的风格") as demo:
    gr.Markdown(" # 点一下瞧瞧😝")

    # 进度条组件
    progress_display = gr.HTML(label="生成进度")

    with gr.Row():
        content_input = gr.Image(label="内容图", type="filepath", height=256, width=256)
        style_input = gr.Image(label="风格图", type="filepath", height=256, width=256)

    submit_btn = gr.Button("开始生成艺术作品", variant="primary")
    output_image = gr.Image(label="实时结果", type="pil", height=256, width=256)

    gr.Examples(
        examples=example_data,
        inputs=[content_input, style_input],
        label="选择一个示例来快速体验"
    )

    submit_btn.click(
        fn=transfer,
        inputs=[content_input, style_input],
        outputs=[progress_display, output_image]
    )

demo.launch()