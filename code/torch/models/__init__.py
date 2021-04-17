from .AlexNet import AlexNet
from .DenseNet201 import DenseNet201
from .InceptionV3 import InceptionV3
from .MobileNetV2 import MobileNetV2
from .ResNet50 import ResNet50
from .ResNet152 import ResNet152
from .SqueezeNet import SqueezeNet
from .VGG16 import VGG16
from .VGG19 import VGG19
from .VGG19bn import VGG19bn


def get_models() -> dict:
    """Dizionario con tutti i modelli disponibili

        Return:
            dict{str: ProtoModel}: dizionario con tutti i modelli disponibili
    """
    
    modelli = {
        'AlexNet': AlexNet(),
        'DenseNet201': DenseNet201(),
        'InceptionV3': InceptionV3(),
        'MobileNetV2': MobileNetV2(),
        'ResNet50': ResNet50(),
        'ResNet152': ResNet152(),
        'SqueezeNet': SqueezeNet(),
        'VGG16': VGG16(),
        'VGG19': VGG19(),
        'VGG19bn': VGG19bn()
    }

    return modelli