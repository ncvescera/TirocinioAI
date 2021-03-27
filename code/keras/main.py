from os import listdir
from sys import argv
import models
from threading import Thread

dataset_dir = '../dataset/imagenet-mini/val'


def worker(model):
    model.test(dataset_dir)


def main(args):
    # prende tutti i modelli disponibili
    modelli = models.get_models()

    ts = []
    for key, model in modelli.items():
        t = Thread(target=worker, args=(model,))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()
    # testing del modello InceptionV3
    # modelli['InceptionV3'].test(dataset_dir)


if __name__ == "__main__":
    main(argv[1:])
