from sys import argv
import models


def main(args):

    models_fun = models.get_models()
    
    '''
    res = models_fun['MobileNet'](args[0])
    print(res)
    '''
    for key, fun in models_fun.items():
        print(key)

        res = fun(args[0])
        print(res)


if __name__ == "__main__":
    main(argv[1:])
