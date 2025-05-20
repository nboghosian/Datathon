import streamlit as st
import pandas as pd
import joblib
from modelo import prever_candidatos_para_vaga  # função que você já tem

# Carregar modelo e base de candidatos
modelo = joblib.load("modelo_xgb_final.pkl")
df_candidatos = pd.read_csv("df_candidatos_tratado.csv")  # já com variáveis derivadas

st.title("Recomendação de Candidatos para Vaga")

# Inputs da vaga
titulo_vaga = st.text_input("Título da Vaga")
competencias = st.text_area("Competências Técnicas e Comportamentais")
nivel_academico = st.selectbox("Nível Acadêmico", ["Ensino Médio", "Superior Incompleto", "Superior Completo", "Pós-graduação", "Mestrado", "Doutorado"])
nivel_ingles = st.selectbox("Nível de Inglês", ["Básico", "Intermediário", "Avançado", "Fluente"])
tipo_contratacao = st.selectbox("Tipo de Contratação", ["CLT", "PJ/Autônomo", "Estágio", "Cooperado"])
local_vaga = st.text_input("Local da vaga (Cidade, Estado)")

# Botão de recomendação
if st.button("Buscar candidatos recomendados"):
    nova_vaga = {
        "titulo_vaga": titulo_vaga,
        "competencia_tecnicas_e_comportamentais": competencias,
        "nivel_academico_y": nivel_academico,
        "nivel_ingles_y": nivel_ingles,
        "tipo_contratacao": tipo_contratacao,
        "local_vaga": local_vaga
    }

    resultados = prever_candidatos_para_vaga(nova_vaga, df_candidatos, modelo, threshold=0.3)

    st.subheader("Candidatos recomendados")
    st.dataframe(resultados)


