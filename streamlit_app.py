import streamlit as st
import pandas as pd
import joblib
from modelo import gerar_variaveis_match

# Carregar modelo e base de candidatos
modelo = joblib.load('modelo_xgb_final.pkl')
df_candidatos = pd.read_csv('df_candidatos_tratado.csv')

st.title("Recomendação de Candidatos para Vaga")

# Inputs da vaga
st.subheader("Dados da Vaga")
titulo_vaga = st.text_input("Título da Vaga")
competencias = st.text_area("Competências Técnicas e Comportamentais")
nivel_academico = st.selectbox("Nível Acadêmico", ["Ensino Médio", "Superior Incompleto", "Superior Completo", "Pós-graduação", "Mestrado", "Doutorado"])
nivel_ingles = st.selectbox("Nível de Inglês", ["Básico", "Intermediário", "Avançado", "Fluente"])
local_vaga = st.text_input("Local da vaga (Cidade, Estado)")

vaga = {
    "titulo_vaga": titulo_vaga,
    "competencia_tecnicas_e_comportamentais": competencias,
    "nivel_academico_y": nivel_academico,
    "nivel_ingles_y": nivel_ingles,
    "local_vaga": local_vaga
}

if st.button("Buscar Candidatos"):
    # Gerar variáveis de match
    df_match = gerar_variaveis_match(df_candidatos, vaga)

    # Aqui você aplicaria o mesmo pré-processamento usado no treino
    X = df_match[[
        'match_competencias',
        'match_titulo_vaga_perfil',
        'match_local',
        'match_ingles',
        'match_nivel_academico',
        'delta_ingles',
        'delta_academico'
    ]]

    # Predição de probabilidade
    proba = modelo.predict_proba(X)[:, 1]
    df_match['prob_contratacao'] = proba

    # Filtrar os melhores
    resultado = df_match.sort_values(by='prob_contratacao', ascending=False)

    st.subheader("Candidatos recomendados:")
    st.dataframe(resultado[['codigo_candidato', 'titulo_profissional', 'prob_contratacao']])
