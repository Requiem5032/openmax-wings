import torch

from pretrainedmodels.models import xception


def Mos_Xception(class_num):
    model = xception(pretrained="imagenet")
    model.last_linear = torch.nn.Linear(
        in_features=2048,
        out_features=class_num,
        bias=True,
    )

    return model
