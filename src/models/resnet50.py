import torch
import torch.nn as nn
from albumentations import Resize, CenterCrop, Normalize, Sequential
from models.model import Model
from pytorchvideo.models import create_res_basic_head


class ResNet50(Model):
    
    def __init__(self, num_classes=1, to_train=0):
        
        super().__init__(num_classes=num_classes, to_train=to_train)
        
        self.preprocessing = Sequential([
                            Resize(height=256, width=256, always_apply=True),
                            CenterCrop(height=256, width=256, always_apply=True),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.,
                                        always_apply=True)                            ])
        
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = True        
        
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        

    def forward(self, x):
        # x must be (batch_size, channels, height, width) or (channels, height, width) to go
        # through the model.

        if len(x.shape) == 5:
            # time dimension is present
            batch_size, time_steps, C, H, W = x.size()

            # reshape input  to be (batch_size * timesteps, C, H, W)
            x = x.contiguous().view(batch_size * time_steps, C, H, W)

        x = self.model(x)
        return x
