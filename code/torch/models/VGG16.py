import torch
import torchvision.models as models
from .ProtoModel import ProtoModel


class VGG16(ProtoModel):
    def __init__(self):
        super().__init__()

        self.name = 'VGG16'
        self.model = None
        self.img_size = (224, 244)

    def _init_model(self):
        self.model = models.vgg16(pretrained=True)

    # classifica l'immagine passata
    # Ritorna una lista contenente 5 dizionari (TOP 5) del tipo:
    #   {'class': 'classe predetta', 'probability': 0.2345134}
    #
    # image: path dell'immagine da classificare
    def predict(self, image: str) -> list:
        # inizializza il modello la prima volta che viene chiamata questa funzione
        if self.model == None:
            self._init_model()
        
        img = self.prepare_image(image, img_size=self.img_size) # prepara l'immagine per essere classificata
        res = self.predict_proto(self.model, img)   # classifica l'immagine

        return res

    def test(self):
        pass