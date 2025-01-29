import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')
lemmatizador = WordNetLemmatizer()

with open('datos.json', encoding='utf-8') as archivo:
    datos_intenciones = json.load(archivo)

with open('words.pkl', 'rb') as archivo:
    vocabulario = pickle.load(archivo)

with open('classes.pkl', 'rb') as archivo:
    categorias = pickle.load(archivo)

vectorizador = CountVectorizer(vocabulary=vocabulario)

X_entrenamiento = []
y_entrenamiento = []

for intencion in datos_intenciones['intents']:
    for ejemplo in intencion['patterns']:
        tokens = nltk.word_tokenize(ejemplo)
        lematizados = [lemmatizador.lemmatize(palabra.lower()) for palabra in tokens]
        bolsa = vectorizador.transform([' '.join(lematizados)]).toarray()[0]
        X_entrenamiento.append(bolsa)
        y_entrenamiento.append(categorias.index(intencion['tag']))

X_entrenamiento = np.array(X_entrenamiento)
y_entrenamiento = np.array(y_entrenamiento)

modelo = MultinomialNB()
modelo.fit(X_entrenamiento, y_entrenamiento)

def procesar_frase(frase):
    """Tokeniza y lematiza una frase."""
    palabras = nltk.word_tokenize(frase)
    return [lemmatizador.lemmatize(p.lower()) for p in palabras]

def generar_bolsa_palabras(frase):
    """Convierte una frase en una bolsa de palabras."""
    palabras_procesadas = procesar_frase(frase)
    return vectorizador.transform([' '.join(palabras_procesadas)]).toarray()[0]

def predecir_intencion(frase):
    """Predice la intención de una frase ingresada."""
    bolsa = generar_bolsa_palabras(frase)
    probabilidades = modelo.predict_proba([bolsa])[0]
    UMBRAL_ERROR = 0.25
    resultados = [[i, prob] for i, prob in enumerate(probabilidades) if prob > UMBRAL_ERROR]
    resultados.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': categorias[r[0]], 'probability': str(r[1])} for r in resultados]

def obtener_respuesta(lista_intenciones, json_intenciones):
    """Devuelve una respuesta basada en la intención detectada."""
    if lista_intenciones:
        etiqueta = lista_intenciones[0]['intent']
        for intencion in json_intenciones['intents']:
            if intencion['tag'] == etiqueta:
                return random.choice(intencion['responses'])
    return "Lo siento, no comprendí tu mensaje."

if __name__ == "__main__":
    print("Chatbot activo. Escribe un mensaje para interactuar (o escribe 'salir' para cerrar).")
    while True:
        entrada_usuario = input("Tú: ")
        if entrada_usuario.lower() == "salir":
            print("Chatbot apagado.")
            break
        lista_intenciones = predecir_intencion(entrada_usuario)
        respuesta = obtener_respuesta(lista_intenciones, datos_intenciones)
        print(f"Bot: {respuesta}")
