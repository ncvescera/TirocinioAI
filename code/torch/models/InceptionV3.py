import torch
import torchvision.models as models
from .ProtoModel import ProtoModel


class InceptionV3(ProtoModel):
    def __init__(self):
        super().__init__()

        self.name = 'InceptionV3'
        self.model = None
        self.img_resize = 299
        self.img_crop = 299

    # funzione per la lateinit del modello
    def _init_model(self):
        self.model = models.inception_v3(pretrained=True)
        self.model.eval()    # disabilita la fase di training e permette di testare il modello

    # classifica l'immagine passata
    # Ritorna una lista contenente 5 dizionari (TOP 5) del tipo:
    #   {'class': 'classe predetta', 'probability': 0.2345134}
    #
    # image: path dell'immagine da classificare
    def predict(self, image: str) -> list:
        # inizializza il modello la prima volta che viene chiamata questa funzione
        if self.model == None:
            self._init_model()
        
        img = self.prepare_image(image, self.img_resize, self.img_crop) # prepara l'immagine per essere classificata
        res = self.predict_proto(self.model, img)   # classifica l'immagine

        return res

    # testa il modello con tutte le immagini del dataset 
    # e scrive un file csv con i risultati
    #
    # dataset_path: percorso del dataset da utilizzare
    def test(self, dataset_path: list):
        self.proto_test(self.name, dataset_path, self.predict)