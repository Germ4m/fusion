import random
import json
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')
lemmatizador = WordNetLemmatizer()

with open('datos.json', encoding='utf-8') as file:
    data = json.load(file)

with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

vectorizer = CountVectorizer(vocabulary=words)

X_train = []
y_train = []

for intent in data['intents']:
    for sentence in intent['patterns']:
        tokens = word_tokenize(sentence)
        lemmatized_words = [lemmatizador.lemmatize(word.lower()) for word in tokens]
        word_bag = vectorizer.transform([' '.join(lemmatized_words)]).toarray()[0]
        X_train.append(word_bag)
        y_train.append(classes.index(intent['tag']))

X_train = np.array(X_train)
y_train = np.array(y_train)

model = MultinomialNB()
model.fit(X_train, y_train)

def preprocess_text(text):
    """Tokeniza y lematiza un mensaje."""
    tokens = word_tokenize(text)
    return [lemmatizador.lemmatize(token.lower()) for token in tokens]

def create_word_bag(text):
    """Transforma un mensaje en su bolsa de palabras."""
    processed_text = preprocess_text(text)
    return vectorizer.transform([' '.join(processed_text)]).toarray()[0]

def predict_intent(text):
    """Predice la intención de un mensaje."""
    word_bag = create_word_bag(text)
    probabilities = model.predict_proba([word_bag])[0]
    THRESHOLD = 0.25
    results = [[i, prob] for i, prob in enumerate(probabilities) if prob > THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intent_list, intents_json):
    """Obtiene una respuesta basada en la intención detectada."""
    if intent_list:
        tag = intent_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "Lo siento, no entendí tu mensaje."

if __name__ == "__main__":
    print("Chatbot activado. Escribe un mensaje para chatear o 'salir' para cerrar.")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("Chatbot desactivado.")
            break
        predicted_intents = predict_intent(user_input)
        response = get_response(predicted_intents, data)
        print(f"Bot: {response}")
