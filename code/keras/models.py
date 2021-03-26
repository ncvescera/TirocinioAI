# disabilita l'output di tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image


# trasforma il risultato della classificazione in un array
# di dizionari pronto per essere utilizzato dagli altri script
def to_dict(arr: list):
    result = []

    for elem in arr:
        tmp_dict = {}

        tmp_dict['probability'] = elem[2]
        tmp_dict['class'] = f'{elem[0]} {elem[1]}'

        result.append(tmp_dict)

    return result


def model_ResNet50(img: str):
    from tensorflow.keras.applications.resnet50 import ResNet50
    #from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    
    model = ResNet50(weights='imagenet')

    img_path = img

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=5)[0])
    
    decoded =  decode_predictions(preds, top=5)[0]
    
    return to_dict(decoded)


def model_InceptionV3(img: str):
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    #from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

    model = InceptionV3(weights='imagenet')

    img_path = img

    img = image.load_img(img_path, target_size=(299, 299))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    
    decoded =  decode_predictions(preds, top=5)[0]
    
    return to_dict(decoded)


def model_MobileNet(img: str):
    from tensorflow.keras.applications.mobilenet import MobileNet
    #from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

    model = MobileNet(weights='imagenet')

    img_path = img

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    
    decoded =  decode_predictions(preds, top=5)[0]
    
    return to_dict(decoded)


def get_models():
    models = {
        'ResNet50': model_ResNet50,
        'InceptionV3': model_InceptionV3,
        'MobileNet': model_MobileNet
    }

    return models
