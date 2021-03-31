from .MobileNetModel import MobileNetModel
from .ResNet50Model import ResNet50Model
from .InceptionV3Model import InceptionV3Model
from .VGG16Model import VGG16Model
from .VGG19Model import VGG19Model
from .ResNet50V2Model import ResNet50V2Model
from .ResNet152V2Model import ResNet152V2Model
from .DenseNet201Model import DenseNet201Model
from .EfficientNetB7Model import EfficientNetB7Model


# ritorna un dizionario con tutti i modelli disponibili
def get_models() -> dict:
    models = {
        'ResNet50': ResNet50Model(),
        'MobileNet': MobileNetModel(),
        'InceptionV3': InceptionV3Model(),
        'VGG16': VGG16Model(),
        'VGG19': VGG19Model(),
        'ResNet50V2': ResNet50V2Model(),
        'ResNet152V2': ResNet152V2Model(),
        'DenseNet201': DenseNet201Model(),
        'EfficientNetB7': EfficientNetB7Model()
    }

    return models
