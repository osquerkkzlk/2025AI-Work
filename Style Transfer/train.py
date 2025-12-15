import torch
from torch import nn
import tqdm
import torchvision
from utils import Recorder,display_loss

loss = nn.MSELoss()

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def get(content_image, style_img, device):
    print("Loading VGG19...")
    net = torchvision.models.vgg19(weights='IMAGENET1K_V1').features.to(device)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    print("Transforming images...")
    content_features = get_features(content_image, net)
    style_features = get_features(style_img, net)
    style_grams = {layer: gram(style_features[layer]) for layer in style_features}
    target = content_image.clone().requires_grad_(True).to(device)

    return content_features, style_grams, target, net

def gram(X):
    b, c, h, w = X.size()
    X = X.view(c, h * w)
    return torch.mm(X, X.t()) / (c * h * w)

def content_loss(y_pred, y):
    return loss(y_pred, y.detach())

def style_loss(y_pred, y):
    return loss(y_pred, y.detach())

def tv_loss(y, weight=1e-4):
    return weight * (torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :]).mean() +
                     torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1]).mean())

def criterion(X, content_val, content_pred, style_gram, style_pred):
    content_weight=1e-4
    style_weight=2e4
    tv_weights=1e-2
    style_weight_assigned = {'conv1_1': 1,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    c_loss = sum([content_loss(yp, y) for yp, y in zip(content_pred, content_val)])
    s_loss = sum([weight*style_loss(gram(yp), gy) for yp, gy,weight in zip(style_pred.values(), style_gram.values(),style_weight_assigned.values())])
    t_loss = tv_loss(X, tv_weights)
    total = content_weight * c_loss + style_weight * s_loss + t_loss
    return c_loss, s_loss, t_loss, total


def train(configue, content_img, style_img, display_loss_button=True):
    device = configue["device"]
    content_X, style_grams, target, net = get(content_img, style_img, device)

    optim=torch.optim.AdamW([target],lr=0.003)
    loss_recorder = Recorder(4)
    img_recorder = Recorder(1)

    pbar = tqdm.tqdm(total=configue["epochs"], desc="Training...")
    for epoch in range(configue["epochs"]):
        target_features = get_features(target, net)
        optim.zero_grad()
        c_loss, s_loss, t_loss, total=criterion(target, content_X["conv4_1"], target_features["conv4_1"],\
                                            style_grams, target_features)
        total.backward()
        optim.step()

        loss_recorder.add(c_loss.item(),s_loss.item(), t_loss.item(),total.item())
        if (epoch+1) % configue["epoch_step"] == 0:
            img_recorder.add(target.cpu().clone().detach())
        pbar.update(1)
        pbar.set_description(f"<{epoch+1}/{configue['epochs']}>")
        pbar.set_postfix(loss=f"{loss_recorder[3][-1]:.4f}")

    if display_loss_button:
        display_loss(loss_recorder[0], loss_recorder[1], loss_recorder[2], loss_recorder[3])

    return target, img_recorder[0]