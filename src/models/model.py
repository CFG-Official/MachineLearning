import torch.nn as nn

class Model(nn.Module):

    __slots__ = ['num_classes', 'to_train', 'preprocessing']
    
    def __init__(self, num_classes=1, to_train=0):
        
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.to_train = to_train

    def forward(self, x): 
        pass
    
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
