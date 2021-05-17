import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from model_training import create_model, prepare_images
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img


MODEL_PATH      = './deep_undeeper224.model' # path del modello allenato
VALIDATION_PATH = './ilsvrc2012Validation'   # path del dataset di validazione (1001 imgs)


def main():
    model = create_model()
    model = keras.models.load_model(MODEL_PATH)

    validation :DirectoryIterator = prepare_images(VALIDATION_PATH, shuffle=False)

    # valutazione del modello
    result_eval = eval(model, validation)
    print(f'\nLoss: {result_eval} ({model.metrics_names})')

    # utilizzo del modello con le immagini di validazione
    result_predict :np.array = model.predict(validation)          # passo al modello le immagini di validazione

    validation_np :np.array = dirIterator_to_array(validation)    # trasforma DirectoryIterator in np.arry per poter essere usato dopo
    
    # salva la prima immagine originale e quella ricreata dal modello
    original = array_to_img(validation_np[0]).convert('RGB')    # forzo la conversione a RGB per evitare problemi
    encoded = array_to_img(result_predict[0]).convert('RGB')    # con eventuali immagini in GrayScale

    original.save('./original.jpeg')
    encoded.save('./encoded.jpeg')

    # calcola l'errore quadratico e altri valori per valutare il modello
    errori          = np.array([])  # array con i risultati finali
    conta_distorta  = 0             # immagine non appartenenti al dataset di origine
    conta_pulita    = 0             # immagini appartenenti al dataset di origine
    treshold        = 0.87          # treshold di attivazione per considerare un'immagine pulita (<= treshold) o distorata (> treshold)
    i               = 0             # contatore per accedere alle posizioni dei 2 array

    for i in range(len(validation_np)):
        errore = (np.square(validation_np[i] - result_predict[i])).mean()   # calcola l'errore (??)
        errore = errore * 100
        errori = np.append(errori, errore)

        # conta quante sono le immagini considerate 'distorte' e quelle 'pulite'
        if errore > treshold :
            conta_distorta += 1
        else:
            conta_pulita += 1
    
    print('')
    print(f"immagini pulite = {conta_pulita} ({conta_pulita/len(validation_np)*100}%)")
    print(f"immagini distorte =  {conta_distorta} ({conta_distorta/len(validation_np)*100}%)")
    print(f"media errore dataset = {np.mean(errori)}")


def eval(model: Model, validation: DirectoryIterator) -> float:
    """ Valuta il modello con le immagini passate

        Parameters:
            model (Model)                   : modello da valutare
            validation (DirectoryIterator)  : set di immagini da utilizzare
        
        Return:
            float: valore loss del modell
    """

    return model.evaluate(validation)


def dirIterator_to_array(iter: DirectoryIterator) -> np.array:
    """ Converte un DirectoryIterator in numpy.arry

        Parameters:
            iter (DirectoryIterator): DirectoryIterator da trasformare in array
        
        Return:
            numpy.array: array con tutte le immagini generate da DirectoryIterator
    """

    data_list = []
    batch_index = 0

    while batch_index <= iter.batch_index:
        data = iter.next()
        tmp = data[0]
        for elem in tmp:
            data_list.append(elem)

        batch_index = batch_index + 1

    data_array = np.array(data_list)

    return data_array


if __name__ == "__main__":
    main()