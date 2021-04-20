from .filters import perpetua, stinson, rise, hudson, slumber
import numpy as np
from os import listdir
from PIL import Image


'''
funzione che applica una sequenza random di filtri
con valori causali dei pramatetri
'''
def random_filters(image, n_filters):
    '''
    Ogni lista della tripla contien:
    - funzione da applicare
    - valore più basso generabile per alpha
    - valore più alto generabile per alpha
    '''
    filters = [
        (hudson, 0.9, 1.2),
        (slumber, 0.9, 1.2),
        (stinson, 0.9, 1.2),
        (perpetua, 0.9, 1.2),
        (rise, 0.9, 1.2)
    ]
    img = image.copy()
    for i in range(n_filters):
        r = np.random.randint(len(filters))
        f, l, h = filters.pop(r)
        # genera il valore random per alpha
        a = np.random.uniform(l,  h)
        # genera il valore random per intensity
        s = np.random.uniform(0.7, 1.0)
        # print(f.__name__, s, a)
        img = f(img, s, a)
    return img


def apply_random_filters(seed: int, number_f=5):
    np.random.seed(seed)    # importa il seed random per ripetibilita' del test

    # numero dei filtri da applicare
    n_filters = number_f

    files = listdir("./Images/")
    for img_name in files:
        image = Image.open("./Images/" + img_name)
        print(img_name)

        # se l'immagine è in scala di grigi la converte in RGB
        if image.mode == "L":
            image = image.convert("RGB")

        filt = random_filters(image, n_filters)
        filt.save("./Filtered/" + img_name)
        # print("\n")
