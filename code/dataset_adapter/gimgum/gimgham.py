import numpy as np
import PIL 
from PIL import Image, ImageEnhance


def _to_pil_image(image: np.array) -> Image:
    """Converte un numpy.array in immagine PILLOW

        Parameters:
            image (numpy.array): array che deve essere convertito

        Return:
            Image: immagine in formato PILLOW
    """

    return PIL.Image.fromarray((image).astype(np.uint8))


def _pil_to_array(image: Image) -> np.array:
    """Converte un immagine PILLOW in numpy.array

        Parameters:
            image (Image): immagine in formato PILLOW

        Return:
            numpy.array: immagine sotto forma di numpy.array
    """

    return np.array(image).astype(np.uint8)


def show_image(image: np.array):
    img = _to_pil_image(image)
    img.show()


def edge_enhance(image: np.array, alpha=0.5) -> np.array:
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Color(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)


def brightness (image: np.array, alpha = 1.0) -> np.array:
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Brightness(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)


def contrast(image: np.array, alpha=1.0) -> np.array:
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Contrast(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)


def interpolate(im1: np.array, im2: np.array, alpha: int) -> np.array:
    """Interpola (unisce) due immagini

        Parameters:
            im1 (numpy.array)   : prima immagine da unire
            im2 (numpy.array)   : seconda immagine da unire
            alpha (int)         : parametro ??

        Return:
            numpy.array: immagine risultate dall'interpolazione delle due immagini
    """

    return ((1.0 - alpha) * im1 + alpha * im2)

def gingham(in_image: np.array, intensity = 1 , alpha = 1) -> np.array:
    """Applica il filtro gingham ad un numpy.array

        Parameters:
            in_image (numpy.array)  : immagine da modificare, in formato numpy.array
            intensity (int)         : parametro ??
            alpha (int)             : parametro ??

        Return:
            numpy.array: immagine modificata con il filtro
    """

    image :np.array     = brightness(in_image, 1.1 * alpha)
    image :np.array     = edge_enhance(image, 1.1 * alpha)
    image :np.array     = contrast(image, 0.7 * alpha)
    out_image :np.array = interpolate(in_image, image, intensity)

    return out_image


def apply_gingham(img_path: str, dest_path='./'):
    """Applica il filtro gimgham all'immagine passata e la salva

        Parameters:
            img_path (str): immagini da modificare
    """

    img :Image          = Image.open(img_path).convert('RGB')   # apertura dell'immagini (forza RGB mode)
    img_array :np.array = _pil_to_array(img)                    # l'immagine viene convertita in un numpy.array per essere usata dalle altre funzioni
    res_array :np.array = gingham(img_array)                    # applica il filtro all'immagine
    res_pil :Image      = _to_pil_image(res_array)              # conversione da array a Pillow Image

    # salvataggio dell'immagine
    img_name :str = img_path.split('/')[-1]                                     # prende il nome del file
    img_dest :str = f'{dest_path}/' if dest_path[-1] != '/' else dest_path     # formatta la destinazione aggiungendo un '/' se manca

    res_pil.save(f'{img_dest}{img_name}')      # salva il file
