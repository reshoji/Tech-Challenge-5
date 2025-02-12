import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
import nltk
from nltk.corpus import stopwords
import string

# Baixar stopwords (apenas uma vez)
nltk.download('stopwords')
# Definir stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Carregar o modelo
model = tf.keras.models.load_model('modelo_bolsaestudo.h5')
model2 = joblib.load('modelo_destaque.pkl')
vectorizer = joblib.load('vectorizer.pkl')

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
    # st.write("Saída do modelo:", predicao)

# Título da aplicação
#st.title("📝 Análise de Sentimento")

#st.write("Digite um texto e veja se o sentimento é positivo ou negativo!")

# Criar caixa de texto para entrada do usuário
#texto = st.text_area("Digite seu texto aqui:", "")

#def preprocessar_texto(texto):
   # """Remove stopwords, pontuações e transforma o texto em minúsculas"""
    #texto = texto.lower()  # Converter para minúsculas
    #texto = texto.translate(str.maketrans('', '', string.punctuation))  # Remover pontuação
    #palavras = texto.split()  # Separar palavras
    #palavras_filtradas = [palavra for palavra in palavras if palavra not in stop_words]  # Remover stopwords
    #return " ".join(palavras_filtradas)  # Rejuntar palavras limpas

    
# Botão de previsão
#if st.button("🔍 Analisar Sentimento"):
    #if texto.strip() == "":
        #st.warning("Por favor, insira um texto para análise.")
    #else:
        # Transformar o texto em uma matriz para o modelo (dependendo do pré-processamento usado)
        # Aplicar pré-processamento no texto
        #texto_processado = preprocessar_texto(texto)
        
        # Transformar o texto processado para o formato esperado pelo modelo
        #dados_transformados = vectorizer.transform([texto_processado]) 
        # Fazer previsão
        #predicao = model2.predict(dados_transformados)
        
        # Interpretar resultado
        #resultado = "😊 Positivo!" if predicao[0] == 1 else "☹️ Negativo!"
        
        # Exibir o resultado
        #st.subheader(f"Resultado: {resultado}")
        #st.write(f"Valor bruto da predição: {predicao}")
