import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
import re
import unidecode
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Carregar o modelo
model = tf.keras.models.load_model('modelo_bolsaestudo.h5')
#model2 = joblib.load('modelo_destaque.pkl')
#vectorizer = joblib.load('vectorizer.pkl')

# Configurar o título da aplicação
st.title("Previsão de Bolsa de Estudos")

st.write("Preencha os 8 índices do aluno para verificar a indicação de bolsa.")

# Criar campos de entrada para os 8 índices
INDE_2022 = st.number_input("INDE", min_value=0.0, max_value=10.0, step=0.1)
IAA_2022 = st.number_input("IAA", min_value=0.0, max_value=10.0, step=0.1)
IEG_2022 = st.number_input("IEG", min_value=0.0, max_value=10.0, step=0.1)
IPS_2022 = st.number_input("IPS", min_value=0.0, max_value=10.0, step=0.1)
IDA_2022 = st.number_input("IDA", min_value=0.0, max_value=10.0, step=0.1)
IPP_2022 = st.number_input("IPP", min_value=0.0, max_value=10.0, step=0.1)
IPV_2022 = st.number_input("IPV", min_value=0.0, max_value=10.0, step=0.1)
IAN_2022 = st.number_input("IAN", min_value=0.0, max_value=10.0, step=0.1)

# Botão de previsão
if st.button("🔍 Simular Bolsa de Estudos"):
    # Criar array com os dados inseridos
    dados_aluno = np.array([[INDE_2022, IAA_2022, IEG_2022, IPS_2022, IDA_2022, IPP_2022, IPV_2022, IAN_2022]])

    # Fazer previsão
    predicao = model.predict(dados_aluno)

    # Mostrar resultado
    resultado = "✅ Aprovado para Bolsa!" if predicao[0][0] >= 0.97 else "❌ Não Aprovado para Bolsa."
    st.subheader(resultado)
    st.write("Saída do modelo:", predicao)





#Tentar carregar o modelo e o vetorizador
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model2 = joblib.load("modelo_destaque.pkl")
except FileNotFoundError:
    st.error("Erro: Arquivos do modelo não encontrados. Verifique os caminhos dos arquivos!")

# Definir o caminho personalizado para os dados do nltk
nltk.data.path.append(r"https://github.com/reshoji/Tech-Challenge-5/tree/main/stopwords")

# Configurar Stopwords e Stemmer
stop_words = set(stopwords.words("portuguese"))
stemmer = SnowballStemmer("portuguese")

def preprocess_text(text):
    """Pré-processa o texto removendo acentos, pontuações e aplicando stemming."""
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

#Interface do Streamlit
st.title("Análise de Sentimento 💬")
st.write("Digite uma frase para analisar se o sentimento é positivo ou negativo.")

#Caixa de entrada para texto
input_text = st.text_area("Digite sua frase aqui:")

if st.button("Analisar"):
    if input_text.strip() == "":
        st.warning("Por favor, digite uma frase para análise.")
    else:
        # Processar o texto
        frase_processada = preprocess_text(input_text)
        frase_tfidf = vectorizer.transform([frase_processada])

#Fazer previsão
        predicao = model2.predict(frase_tfidf)[0]
        sentimento = "😊 Positivo" if predicao == 1 else "😞 Negativo"

#Exibir resultado
        st.success(f"Resultado: {sentimento}")
