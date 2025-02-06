import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# Carregar o modelo
model = tf.keras.models.load_model('modelo_bolsaestudo.h5')

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

