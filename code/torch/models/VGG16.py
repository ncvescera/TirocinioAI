import torch
import torchvision.models as models
from .ProtoModel import ProtoModel


class VGG16(ProtoModel):
    def __init__(self):
        super().__init__()

        self.name = 'VGG16'
        self.model = None
        self.img_resize = 256
        self.img_crop = 224

    def _init_model(self):
        """Funzione per la lateinit del modello
        """

        self.model = models.vgg16(pretrained=True)
        self.model.eval()    # disabilita la fase di training e permette di testare il modello

    def predict(self, image: str, grayScale=False) -> list:
        """Classifica l'immagine passata
            
            Parameters:
                image (str)     : path dell'immagine da classificare
                grayScale (bool): se 'True', applica alle immagini il filtro GrayScale

            Return:
            list[dict{'class': str, 'probability': double}]: 
                Ritorna una lista contenente 5 dizionari (TOP 5) del tipo:
            
                {'class': 'classe predetta', 'probability': 0.2345134}
        """

        # inizializza il modello la prima volta che viene chiamata questa funzione
        if self.model == None:
            self._init_model()
        
        img = self.prepare_image(image, self.img_resize, self.img_crop, grayScale=grayScale) # prepara l'immagine per essere classificata
        res = self.predict_proto(self.model, img)                                            # classifica l'immagine

        return res

    def test(self, dataset_path: str, grayScale=False):
        """Testa il modello con tutte le immagini del dataset e scrive un file csv con i risultati

            Parameters:
                dataset_path (str)  : percorso del dataset da utilizzare
                grayScale (bool)    : se 'True', applica alle immagini il filtro GrayScale
        """

        self.proto_test(self.name, dataset_path, self.predict, grayScale=grayScale)