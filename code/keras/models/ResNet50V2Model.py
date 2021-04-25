from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from .ProtoModel import ProtoModel


class ResNet50V2Model(ProtoModel):
    """Classe che implementa le funzioni del modello ResNet50V2
    """

    def __init__(self):
        super().__init__()

        self.model = None
        self.name = 'ResNet50V2'

    def _init_model(self):
        """Lateinit del modello per evitare un avvio mooolto lento
        """

        self.model = ResNet50V2(weights='imagenet')

    def predict(self, imgs: list, gscale=False) -> list:
        """Classifica una lista di immagini date
    
            Parameters:
                imgs (list)     : lista di immagini da classificare
                gscale (bool)   : se 'True', applica il filtro GrayScale alle immagini

            Return:
                list[dict{'class': str, 'probability': double}]: 
                    ritorna una lista contenente dizionari del tipo:
                        {'class': 'classe predetta', 'probability': 0.2348172}
                    Ogni elemento della lista è una predizione: contiene i primi 5 risultati (TOP 5)
                    Ci saranno tanti elementi nella lista finali quante immagini
        """

        target_size = (224, 224)    # indica la dimenzione con cui deve essere ridimenzionata l'immagine

        # controllo per il lateinit del modello
        if self.model == None:
            self._init_model()

        result = self.proto_predict(imgs, target_size, self.model, preprocess_input, decode_predictions, gscale) 
        
        return result 
    
    def test(self, dataset_path: str,  gscale=False):
        """Testa il modello con tutte le immagini del dataset passatogli

            Parameters:
                dataset_path (str): il path della cartella contenente il dataset
                gscale (bool): se 'True' alle immagini viene applicato il filtro GrayScale
        """

        self.proto_test(dataset_path, self.name, self.predict, gscale)