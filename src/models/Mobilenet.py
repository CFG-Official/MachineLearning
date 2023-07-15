from models.model import Model
from albumentations import Resize, CenterCrop, Normalize, Sequential
from torch import nn
from torchvision.models import mobilenet_v2

class MobileNet(Model):
 
    def __init__(self, num_classes=1, to_train=0):
        
        super().__init__(num_classes=num_classes, to_train=to_train)
        
        self.preprocessing = Sequential([
                            Resize(height=256, width=256, always_apply=True),
                            CenterCrop(height=256, width=256, always_apply=True),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.,
                                        always_apply=True),
                            ])
        
        if self.to_train is None:
            self.model = mobilenet_v2(pretrained=False)
        else:
            self.model = mobilenet_v2(pretrained=True)

        if self.to_train >= 0: 
            # If a positive number is given, we train the last n blocks of feature extraction
            trainable_features = []
            for i in range(1, self.to_train+1):
                trainable_features.append(self.model.features[-i])

            for feature in trainable_features:
                for param in feature.parameters():
                    param.requires_grad = True
        else:
            trainable_features = self.model.features
            # If a negative number is given, we train all layers
            for feature in trainable_features:
                for param in feature.parameters():
                    param.requires_grad = True

        self.model.classifier[-1] = nn.Linear(1280, self.num_classes)
        self.model.classifier.requires_grad_(True)

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

        