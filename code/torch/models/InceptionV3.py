import torch
import torchvision.models as models
from .ProtoModel import ProtoModel


class InceptionV3(ProtoModel):
    def __init__(self):
        super().__init__()

        self.name = 'InceptionV3'
        self.model = None
        self.img_size = (299, 299)

    def _init_model(self):
        self.model = models.inception_v3(pretrained=True)

    def predict(self, image: str):
        # inizializza il modello la prima volta che viene chiamata questa funzione
        if self.model == None:
            self._init_model()
        
        img = self.prepare_image(image, img_size=self.img_size) # prepara l'immagine per essere classificata
        res = self.predict_proto(self.model, img)   # classifica l'immagine

        return res

    def test(self):
        pass