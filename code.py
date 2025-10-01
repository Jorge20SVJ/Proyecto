"""
Phishing Detection Algorithm using Naive Bayes
Text classification system for detecting spam/phishing attempts
"""
u
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Paso 1: Leer el archivo .txt que contiene los mensajes
archivo_txt = 'data/SMSSpamCollection.txt'  # Cambia esto por la ruta de tu archivo
df = pd.read_csv(archivo_txt, sep='\t', names=["classification", "message"])

# Paso 2: Generar etiquetas automáticas
df['target'] = np.where(df['classification'] == 'ham', 0, 1)

# Paso 3: Convertir los textos en vectores numéricos (usando TF-IDF)
vectorizador = TfidfVectorizer(stop_words='english')
X = vectorizador.fit_transform(df['message'])  # Convertir los mensajes a vectores numéricos
y = df['target']  # Etiquetas (1 = phishing, 0 = no phishing)

# Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Paso 5: Entrenar el modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Paso 6: Hacer predicciones con los datos de prueba
y_predicciones = modelo.predict(X_test)

# Paso 7: Traducir y mostrar el informe de clasificación en español
informe = classification_report(y_test, y_predicciones, output_dict=True)

# Mostrar los resultados en español
print(f"Precisión general (Exactitud): {accuracy_score(y_test, y_predicciones):.2f}\n")

# Recorrer cada clase en el informe y traducirlo
for clase, valores in informe.items():
    if clase in ['accuracy', 'macro avg', 'weighted avg']:
        # Traducir las etiquetas generales
        if clase == 'accuracy':
            clase_traducida = 'Precisión (Exactitud global)'
        elif clase == 'macro avg':
            clase_traducida = 'Promedio Macro'
        elif clase == 'weighted avg':
            clase_traducida = 'Promedio Ponderado'
    else:
        clase_traducida = f"Clase {clase}"  # Para las clases numéricas (0 y 1)

    # Formatear y traducir las métricas
    if clase != 'accuracy':
        print(f"{clase_traducida}:")
        print(f"  Precisión: {valores['precision']:.2f}")
        print(f"  Exhaustividad (Recall): {valores['recall']:.2f}")
        print(f"  Puntuación F1: {valores['f1-score']:.2f}")
        print(f"  Soporte: {valores['support']}\n")

# Paso 8: Guardar el modelo y el vectorizador para usarlos más tarde
joblib.dump(modelo, 'modelo_phishing.pkl')
joblib.dump(vectorizador, 'tfidf_vectorizer.pkl')
