from .efficientnet_LSTM_temporal import Efficientnet_LSTM_temporal
from .efficientnet_LSTM_temporal2 import Efficientnet_LSTM_temporal2
from .I3D_R50 import I3D_R50
from .mobilenet_LSTM_temporal import Mobilenet_v2_LSTM_temporal
from .mobilenet_LSTM_temporal2 import Mobilenet_v2_LSTM_temporal2
from .mobilenet_LSTM_temporal2_old import Mobilenet_v2_LSTM_temporal2_old
from .resnet_LSTM_temporal import Resnet18_LSTM_temporal
from .resnet_LSTM_temporal2 import Resnet18_LSTM_temporal2
from .SlowFast import SlowFast
from .Slow_R50 import Slow_R50
from .vgg_LSTM_temporal import VGG16_LSTM_temporal
from .vgg_LSTM_temporal2 import VGG16_LSTM_temporal2
from .X3D_xs import X3D_xs
from .YOLO_temporal2 import YOLO_temporal2
from .X3D_s import X3D_s

class FireDetectionModelFactory:

    models = {
        'efficientnet_LSTM_temporal': Efficientnet_LSTM_temporal,
        'efficientnet_LSTM_temporal2': Efficientnet_LSTM_temporal2,
        'i3d_r50': I3D_R50,
        'mobilenet_v2_LSTM_temporal': Mobilenet_v2_LSTM_temporal,
        'mobilenet_v2_LSTM_temporal2': Mobilenet_v2_LSTM_temporal2,
        'mobilenet_v2_LSTM_temporal2_old': Mobilenet_v2_LSTM_temporal2_old,
        'resnet18_LSTM_temporal': Resnet18_LSTM_temporal,
        'resnet18_LSTM_temporal2': Resnet18_LSTM_temporal2,
        'slowfast': SlowFast,
        'slow_r50': Slow_R50,
        'vgg16_LSTM_temporal': VGG16_LSTM_temporal,
        'vgg16_LSTM_temporal2': VGG16_LSTM_temporal2,
        'x3d_xs': X3D_xs,
        'YOLO_temporal2': YOLO_temporal2,
        'x3d_s': X3D_s
    }

    @staticmethod
    def create_model(model_name, num_classes, to_train):
        if model_name not in FireDetectionModelFactory.models.keys():
            raise ValueError(f"Model '{model_name}' is not supported.")
        model_class = FireDetectionModelFactory.models[model_name]
        return model_class(num_classes, to_train)
