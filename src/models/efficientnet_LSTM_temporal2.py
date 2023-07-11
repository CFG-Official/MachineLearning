from models.model import Model
from albumentations import Resize, CenterCrop, Normalize, Sequential
import torch
from torch import nn
import torchvision.models as models
from torchvision.models import efficientnet_b0

class Efficientnet_LSTM_temporal2(Model):
    
    def __init__(self, num_classes=1, to_train=0, state_dict=None):
        
        super().__init__(num_classes=num_classes, to_train=to_train)
        
        self.preprocessing = Sequential([
                            Resize(height=256, width=256, always_apply=True),
                            CenterCrop(height=224, width=224, always_apply=True),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.,
                                        always_apply=True),
                            ])
        if state_dict is not None:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=self.num_classes, bias=True)
        
            model.classifier.requires_grad = True
            model.load_state_dict(state_dict)
        else:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
                    
        for param in model.parameters():
            param.requires_grad = False
            
        if self.to_train > 0:
            trainable_features = []
            for i in range(1, self.to_train+1):
                trainable_features.append(model.features[-i])

            for feature in trainable_features:
                for param in feature.parameters():
                    param.requires_grad = True
        

        out_features = model.classifier[1].in_features

        self.cnn = model.features
        self.rnn = nn.LSTM(input_size=out_features*64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = torch.nn.Sequential([
            torch.nn.Dropout(p=0.4, inplace=True),
            torch.nn.Linear(in_features=64, out_features=self.num_classes, bias=True)
        ])   

    def forward(self, x):
        # x: (batch_size, time_steps, C, H, W)
        
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