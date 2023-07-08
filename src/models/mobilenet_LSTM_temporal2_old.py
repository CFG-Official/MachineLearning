from models.model import Model
from albumentations import Resize, CenterCrop, Normalize, Sequential
import torch
from torch import nn
from torchvision.models import mobilenet_v2

class Mobilenet_v2_LSTM_temporal2_old(Model):
    
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
            model = mobilenet_v2(pretrained=False)
        else:
            model = mobilenet_v2(pretrained=True)
        
        out_features = model.classifier[1].in_features
        
        for param in model.parameters():
            param.requires_grad = False
            
        if self.to_train > 0:
            trainable_features = []
            for i in range(1, self.to_train+1):
                trainable_features.append(model.features[-i])

            for feature in trainable_features:
                for param in feature.parameters():
                    param.requires_grad = True
        
        model.classifier[-1] = nn.Linear(64, 1)
        
        self.cnn = model.features
        self.rnn = nn.LSTM(input_size=out_features*49, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = model.classifier

    def forward(self, x):
        
        _, time_steps, _, _, _ = x.size()
        
        fatures_all_frames = []
        for i in range(time_steps):
            # CNN
            temp_x = self.cnn(x[:, i, :, :, :])
            # Flatten
            temp_x = temp_x.view(temp_x.size(0), -1)
            fatures_all_frames.append(temp_x)
        
        # stack features of all frames
        c_out = torch.stack(fatures_all_frames, dim=0).transpose_(0, 1)
        fatures_all_frames = None
        temp_x = None
        
        # RNN
        r_out, _ = self.rnn(c_out)
        
        # FC
        out = self.fc(r_out[:, -1, :])
        
        return out