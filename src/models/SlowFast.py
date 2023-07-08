import torch
import torch.nn as nn
from models.model import Model
import albumentations

from pytorchvideo.models import create_res_basic_head


class SlowFast(Model):
    
    def __init__(self, num_classes=1, to_train=0):
        
        super().__init__(num_classes=num_classes, to_train=to_train)

        # Preprocessing parameters
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        
        self.preprocessing = albumentations.Sequential([
            albumentations.Resize(height=side_size, width=side_size),  # Ridimensionamento
            albumentations.CenterCrop(height=crop_size, width=crop_size),  # Ritaglio centrale
            albumentations.Normalize(mean=mean, std=std),  # Normalizzazione dei pixel
        ])

        # Model parameters
        self.alpha = 4
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)

        # Replace the last layer for finetuning
        self.model.blocks[:-to_train].requires_grad_(False)
        self.model.blocks[-1].proj.out_features = num_classes
        
    def __pack_pathway(self, x):
        fast_pathway = x
        linspace = torch.linspace(
                0, x.shape[1] - 1, x.shape[1] // (self.alpha)
            ).long()
        linspace = linspace.to(x.device)
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            x,
            1,
            linspace,
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

    def forward(self, x):
        # x è un batch di frame di video: B x T x C x H x W oppure T x C x H x W
        # B = batch size, T = numero di frame, C = numero di canali, H = altezza, W = larghezza
        if len(x.size()) == 4:
            x = x.unsqueeze(0)

        x = x.permute(0, 2, 1, 3, 4) # B x C x T x H x W

        batch_size, _, _, _, _ = x.size()

        input = []
        
        for i in range(batch_size):
            slow_fast = self.__pack_pathway(x[i])
            slow_fast = [way.unsqueeze(0) for way in slow_fast] # [1xTxCxHxW, 1x(T/alpha)xCxHxW]
            input.append(slow_fast)

        # stack the first element of each list in input
        slow_input = torch.cat([input[i][0] for i in range(batch_size)], dim=0)
        fast_input = torch.cat([input[i][1] for i in range(batch_size)], dim=0)
        
        out = self.model([slow_input, fast_input])
        out = torch.sum(out , dim=1)
        
        return out