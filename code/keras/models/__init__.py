from .MobileNetModel import MobileNetModel
from .ResNet50Model import ResNet50Model
from .InceptionV3Model import InceptionV3Model


def get_models():
    models = {
        'ResNet50': ResNet50Model(),
        'MobileNet': MobileNetModel(),
        'InceptionV3': InceptionV3Model()
    }

    return models