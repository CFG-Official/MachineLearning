from models.model import Model
from albumentations import Resize, CenterCrop, Normalize, Sequential
import torch
from torch import nn
from torchvision import models, transforms
import cv2


class EfficientNet(Model):

    def __init__(self, num_classes=1, to_train=0):
        
        super().__init__(num_classes=num_classes, to_train=to_train)

        self.preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=self.num_classes, bias=True)
        
        self.model.classifier.requires_grad = True
        if self.to_train > 0:
            trainable_features = []
            for i in range(1, self.to_train+1):
                trainable_features.append(self.model.features[-i])

            for feature in trainable_features:
                for param in feature.parameters():
                    param.requires_grad = True
        

    def forward(self, x):
        x = self.model(x)
        return x