import argparse
from filters import hudson, slumber, stinson, perpetua, rise
from PIL import Image

'''

Script per usare i filtri
Parametri:
-i per specificare percorso dell'immagine di input
-o per secificare percorso e nome dell'immaggine di output
-f per passare i filtri da applicare
I filtri applicabili sono hudson, perpetua, rise, slumber e stinson
I filtri vengono applicati nell'ordine nei quali vengono passati
Bisogna per impostare i valori di alpha e intensity vanno passati dopo il nome del filtro (prima intensity e poi alpha)
Se non vengon inseriti i valori di alpha e intensity si assume che sono entrambi pari a 1

Esempio:
python main.py -i Image/image.jpg -f hudson rise 0.7 1.2 perpetua -o Image/output.jpg

Applica all'immagine image.jpg nella cartella Image nell'odrine i filtri hudson
con 1 di intensity e 1 di alpha, rise con 0.7 di intensity e 1.2 di alpha, perpetua
1 di intensity 1 di alpha e salva l'immagine filtrata nella cartella Image con nome output.jpg

'''


# torna true se il valore passato è un numero intero o reale
def is_a_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":

    # definizione degli argomenti da passare al comando
    parser = argparse.ArgumentParser(description="Some Instagram filters")
    parser.add_argument('--input', '-i', type=str, nargs=1, default=None, required=True)
    parser.add_argument('--filters', '-f', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--output', '-o', type=str, nargs=1, default="output.jpg", required=False)
    args = vars(parser.parse_args())

    # dizionario che mappa la funzione che applica il filtro al suo nome
    filters = {
        "hudson": hudson,
        "slumber": slumber,
        "stinson": stinson,
        "perpetua": perpetua,
        "rise": rise
    }

    # estrae dagli argomenti il path di input e output
    in_image_path = args["input"][0]
    out_image_path = args["output"][0] if type(args["output"]) == list else args["output"]

    # estrae dagli argomenti i filtri da applicare con i parametri inseriti
    # se viene inserito solo il nome del filtro allora si assume che alpha
    # e intensity siano nulli
    inserted_filtes = args["filters"]
    selected_filters = []
    temp = []
    for elm in inserted_filtes:
        if not is_a_number(elm):
            if len(temp) == 1:
                temp.append(1.0)
                temp.append(1.0)
            selected_filters.append(temp)
            temp = []
        temp.append(float(elm) if is_a_number(elm) else elm)
    if temp != []:
        if len(temp) == 1:
            temp.append(1.0)
            temp.append(1.0)
        selected_filters.append(temp)
    selected_filters.pop(0)
    

    image = Image.open(in_image_path)

    # applica i filtri all'immagine
    for elm in selected_filters:
        f_name, s, a = elm
        print(f_name, s, a)
        f = filters[f_name]
        image = f(image, s, a)
        

    image.save(out_image_path)


    

