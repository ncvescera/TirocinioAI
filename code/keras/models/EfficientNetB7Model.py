import tensorflow
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from .utils import to_dict
from .utils import get_all_dirs
from .utils import get_all_dirs_files
from .utils import PredictionData
from .utils import save_csv


class EfficientNetB7Model:
    def __init__(self):
        self.model = EfficientNetB7(weights='imagenet')
        self.name = 'EfficientNetB7'
    
    def predict(self, imgs: list):
        results = []
        for img in imgs:
            img_path = img

            img = image.load_img(img_path, target_size=(600, 600))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = self.model.predict(x)
            
            # print(preds)
            decoded =  decode_predictions(preds, top=5)[0]

            results.append(to_dict(decoded))
        
        return results
    
    def test(self, dataset_path: str):
        all_classes = get_all_dirs(dataset_path)
        
        # classifica tutti i file (divisi per classe) del dataset
        predictions = []
        for clas in all_classes:
            print(f'**** {self.name.upper()}  {clas.upper()} ****')

            for image in get_all_dirs_files(clas):
                res = self.predict([image])[0]
                
                # print('. ', end='')

                pred = PredictionData(self.name, image, clas, res)
                predictions.append(pred)
            
            # print()

        save_csv(self.name, predictions)
       