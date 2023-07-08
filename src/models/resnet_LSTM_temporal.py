import torch
import torch.nn as nn
from models.model import Model
from albumentations import Resize, CenterCrop, Normalize, Sequential
from torchvision.models import resnet18

class Resnet18_LSTM_temporal(Model):
    
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
            model = resnet18(pretrained=False)
        else:
            model = resnet18(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        self.cnn = nn.Sequential(*list(model.children())[:-1])
            
        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        
        batch_size, time_steps, C, H, W = x.size()
        
        # reshape input  to be (batch_size * timesteps, input_size)
        x = x.contiguous().view(batch_size * time_steps, C, H, W)
        
        # CNN
        c_out = self.cnn(x)
        # remove the last two dims which are 1
        c_out = c_out.view(c_out.size(0), -1)
        # make output as  ( samples, timesteps, output_size)
        c_out = c_out.contiguous().view(batch_size , time_steps , c_out.size(-1))
        
        # RNN
        r_out, _ = self.rnn(c_out)
        
        # FC
        out = self.fc(r_out[:, -1, :])
        
        return out

    
    