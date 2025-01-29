import json
import pickle
import random
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

nltk.download('wordnet')

with open('datos.json', encoding='utf-8') as archivo:
    datos = json.load(archivo)

lemmatizador = WordNetLemmatizer()

vocabulario = []
categorias = []
corpus = []
simbolos_excluir = {'?', '!'}

for elemento in datos['intents']:
    for oracion in elemento['patterns']:
        palabras_tokenizadas = word_tokenize(oracion)
        vocabulario.extend(palabras_tokenizadas)
        corpus.append((palabras_tokenizadas, elemento['tag']))
        if elemento['tag'] not in categorias:
            categorias.append(elemento['tag'])

vocabulario = sorted(set(lemmatizador.lemmatize(palabra.lower()) for palabra in vocabulario if palabra not in simbolos_excluir))
categorias = sorted(categorias)

pickle.dump(vocabulario, open('words.pkl', 'wb'))
pickle.dump(categorias, open('classes.pkl', 'wb'))

datos_entrenamiento = []
salida_neutra = [0] * len(categorias)

for palabras, etiqueta in corpus:
    bolsa_palabras = [1 if lemmatizador.lemmatize(p.lower()) in palabras else 0 for p in vocabulario]
    resultado = salida_neutra[:]
    resultado[categorias.index(etiqueta)] = 1
    datos_entrenamiento.append([bolsa_palabras, resultado])

random.shuffle(datos_entrenamiento)

X = np.array([fila[0] for fila in datos_entrenamiento])
y = np.array([fila[1] for fila in datos_entrenamiento])

binarizador = LabelBinarizer()
y = binarizador.fit_transform([np.argmax(fila) for fila in y])

X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_red = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=200,
    batch_size=5,
    verbose=True,
    random_state=42
)
modelo_red.fit(X_entrenar, y_entrenar)

joblib.dump(modelo_red, 'modelo_chat.pkl')

print('Modelo entrenado y almacenado con éxito.')

exactitud = modelo_red.score(X_probar, y_probar)
print(f'Precisión del modelo: {exactitud * 100:.2f}%')
