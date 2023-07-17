import torch.nn as nn
from albumentations import Resize, CenterCrop, Normalize, Sequential
from pytorchvideo.models.hub import c2d_r50
from pytorchvideo.models import create_res_basic_head
from models.model import Model

class C2D_R50(Model):
    
    def __init__(self, num_classes=1, to_train=0):

        self.preprocessing = Sequential([
                            Resize(height=256, width=256, always_apply=True),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.,
                                        always_apply=True),
                            ])

        self.model = c2d_r50(pretrained=False)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        if self.to_train > 0:
            trainable_blocks = []
            # train the last to_train blocks
            for i in range(1, self.to_train+1):
                trainable_blocks.append(self.model.blocks[-i])

            for block in trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = True
        
        self.model.blocks[-1] = create_res_basic_head(in_features=2048, out_features=self.num_classes, pool_kernel_size=(1, 7, 7))
        self.model.fc = nn.Linear(self.num_classes, self.num_classes)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)
        