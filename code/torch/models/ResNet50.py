import torch
import torchvision.models as models
from .ProtoModel import ProtoModel


class ResNet50(ProtoModel):
    def __init__(self):
        super().__init__()

        self.name = 'ResNet50'
        self.model = None
        self.img_size = (224, 244)

    # funzione per la lateinit del modello
    def _init_model(self):
        self.model = models.resnet50(pretrained=True)

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

    # testa il modello con tutte le immagini del dataset 
    # e scrive un file csv con i risultati
    #
    # dataset_path: percorso del dataset da utilizzare
    def test(self, dataset_path: list):
        self.proto_test(self.name, dataset_path, self.predict)