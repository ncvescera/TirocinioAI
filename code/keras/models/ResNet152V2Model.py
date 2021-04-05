from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from .ProtoModel import ProtoModel


# Classe che implementa le funzioni del modello ResNet152V2
class ResNet152V2Model(ProtoModel):
    def __init__(self):
        super().__init__()

        self.model = None
        self.name = 'ResNet152V2'

    # lateinit del modello per evitare un avvio mooolto lento
    def _init_model(self):
        self.model = ResNet152V2(weights='imagenet')

    # classifica una lista di immagini date
    # Ritorna una lista contenente dizionari del tipo:
    #   {'class': 'classe predetta', 'probability': 0.2348172}
    # Ogni elemento della lista è una predizione: contiene i primi 5 risultati (TOP 5)
    # Ci saranno tanti elementi nella lista finali quante immagini
    #
    # gscale: se 'True' le immagini verranno convertite in GrayScale
    def predict(self, imgs: list, gscale=False) -> list:
        target_size = (224, 224)    # indica la dimenzione con cui deve essere ridimenzionata l'immagine

        # controllo per il lateinit del modello
        if self.model == None:
            self._init_model()

        result = self.proto_predict(imgs, target_size, self.model, preprocess_input, decode_predictions, gscale) 
        
        return result 
    
    # testa il modello con tutte le immagini del dataset passatogli
    def test(self, dataset_path: str, gscale=False):
        self.proto_test(dataset_path, self.name, self.predict, gscale)