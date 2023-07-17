import torch.nn as nn
from albumentations import Resize, CenterCrop, Normalize, Sequential
from pytorchvideo.models.hub import i3d_r50
from pytorchvideo.models import create_res_basic_head
from models.model import Model

class I3D_R50(Model):
    
    def __init__(self, num_classes=1, to_train=0):
        
        super().__init__(num_classes=num_classes, to_train=to_train)
        
        self.preprocessing = Sequential([
                            Resize(height=256, width=256, always_apply=True),
                            CenterCrop(height=224, width=224, always_apply=True),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.,
                                        always_apply=True),
                            ])

        if self.to_train is None:
            model = i3d_r50(pretrained=False)
        else:
            model = i3d_r50(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
            
        if self.to_train > 0:
            trainable_blocks = []
            # train the last to_train blocks
            for i in range(1, self.to_train+1):
                trainable_blocks.append(model.blocks[-i])

            for block in trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = True
        
        model.blocks[-1] = create_res_basic_head(in_features=2048, out_features=self.num_classes, pool_kernel_size=(1, 7, 7))
        model.fc = nn.Linear(self.num_classes, self.num_classes)
        
        self.model = model

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)