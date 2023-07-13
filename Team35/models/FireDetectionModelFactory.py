from .I3D_R50 import I3D_R50
from .SlowFast import SlowFast
from .Slow_R50 import Slow_R50
from .X3D_xs import X3D_xs
from .X3D_s import X3D_s
from .X3D_l import X3D_l

class FireDetectionModelFactory:

    models = {
        'i3d_r50': I3D_R50,
        'slowfast': SlowFast,
        'slow_r50': Slow_R50,
        'x3d_xs': X3D_xs,
        'x3d_s': X3D_s,
        'x3d_l': X3D_l        
    }

    @staticmethod
    def create_model(model_name, num_classes, to_train):
        if model_name not in FireDetectionModelFactory.models.keys():
            raise ValueError(f"Model '{model_name}' is not supported.")
        model_class = FireDetectionModelFactory.models[model_name]
        return model_class(num_classes, to_train)
