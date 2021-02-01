import timm
import torch
from models import cnn,mlp,lstm
import time

MODELS = {
    "cnn":cnn.CNN,
    "mlp":mlp.Model,
    "lstm":lstm.Model,
    # "mobilenetv2":timm.create_model("mobilenetv2_100",pretrained=False),
    # "resnet50":timm.create_model("resnet50",pretrained=False),
    # "efficientnet_lite":timm.create_model("tf_efficientnet_lite4",pretrained=False)
}




# model = MODELS["cnn"]()
# input = torch.rand((64, 3, 24, 24))
# tic = time.time()
# out = model(input)
# toc = time.time()
# print((toc - tic)*1000)
#
# model.zero_grad()
# loss = torch.randn(64, 10)
# tic = time.time()
# out.backward(loss)
# toc = time.time()
# print((toc - tic)*1000)