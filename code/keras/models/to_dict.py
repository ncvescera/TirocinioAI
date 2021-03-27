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