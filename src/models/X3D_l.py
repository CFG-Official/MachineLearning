import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_l
import albumentations
from models.model import Model
from pytorchvideo.models import create_res_basic_head

class X3D_l(Model):
    
    def __init__(self, num_classes=1, to_train=0):
        
        super().__init__(num_classes=num_classes, to_train=to_train)
        # Preprocessing parameters
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        self.preprocessing = albumentations.Sequential([
            albumentations.SmallestMaxSize(side_size, always_apply=True),
            albumentations.CenterCrop(crop_size, crop_size, always_apply=True),
            albumentations.Normalize(mean=mean,
                                        std=std,
                                        max_pixel_value=255.,
                                        always_apply=True), 
        ])
        # Model parameters
        if self.to_train is None:
            self.model = x3d_l(pretrained=False)
        else:
            self.model = x3d_l(pretrained=True)

        # Replace the last layer for finetuning
        self.model.blocks[:-to_train].requires_grad_(False)
        self.model.blocks[-1] = create_res_basic_head(in_features=192, out_features=num_classes)#, pool_kernel_size=(1, 6, 6))

    def forward(self, x):
        # x è un batch di frame di video: B x T x C x H x W oppure T x C x H x W
        # B = batch size, T = numero di frame, C = numero di canali, H = altezza, W = larghezza
        if len(x.size()) == 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4)

        return self.model(x)