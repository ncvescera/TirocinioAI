# Dati Classificazione

Questi sono i dati grezzi dei vari modelli testati con il dataset ImageNetMini(1000)

I file csv sono formati nel seguente modo:

```
image ; probability ; guess_class ; real_class ; other_predictions
```

dove:

* `image` e' l'immagine passata al modello
* `probability` e' la precisione con cui ha classificato la foto
* `guess_class` e' la classe con cui ha classificato l'immagine. E' formata nel seguente modo: `nxxxxxxxx name1, name2, ...`
* `real_class` e' l'a classe con cui dovrebbe essere classificata l'immagine (presa dal dataset)
* `other_predictions` sono le 5 predizioni che il modello ha fatto (utile per calcolare il TOP 5%)

**N.B.**: nei file reali non ci sono spazi nè nell'header nè nei dati. L'intestazione riportata sopra è scritta con gli spazi solo per renderla più leggibile e chiara
