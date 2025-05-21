import streamlit as st
import pandas as pd
import joblib
from modelo import gerar_variaveis_match

#  Carregar modelo e dados
modelo = joblib.load('modelo_xgb_final.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')  # 🔥 As colunas usadas no treino
df_candidatos = pd.read_csv('df_candidatos_tratado.csv')

st.title("🔍 Recomendação de Candidatos para Vaga")

# Inputs da vaga

st.subheader("📄 Dados da Vaga")

titulo_vaga = st.text_input("Título da Vaga")

senioridade = st.selectbox(
    "Nível de Senioridade", 
    ["Estagiário", "Auxiliar", "Assistente", "Júnior", "Pleno", "Sênior", "Especialista", "Coordenador", "Gerente", "Supervisor"]
)

area_atuacao = st.selectbox(
    "Área de Atuação", 
    ["Desenvolvimento", "Dados", "Governança", "Relacionamento", "Infraestrutura", 
     "Negócio/ADM", "Projetos", "Qualidade", "SAP", "Segurança", "UX", "Outros"]
)

competencias = st.text_area("Competências Técnicas e Comportamentais")

nivel_academico = st.selectbox(
    "Nível Acadêmico",
    [
        'Ensino fundamental incompleto', 'Ensino fundamental cursando', 'Ensino fundamental completo',
        'Ensino médio incompleto', 'Ensino médio cursando', 'Ensino médio completo',
        'Ensino técnico incompleto', 'Ensino técnico cursando', 'Ensino técnico completo',
        'Ensino superior incompleto', 'Ensino superior cursando', 'Ensino superior completo',
        'Pós graduação incompleto', 'Pós graduação cursando', 'Pós graduação completo',
        'Mestrado cursando', 'Mestrado incompleto', 'Mestrado completo',
        'Doutorado cursando', 'Doutorado incompleto', 'Doutorado completo'
    ]
)

nivel_ingles = st.selectbox(
    "Nível de Inglês", 
    ["Básico", "Intermediário", "Avançado", "Fluente"]
)

local_vaga = st.text_input("Local da vaga (Cidade, Estado)")


#  Montar dicionário da vaga
vaga = {
    'titulo_vaga': titulo_vaga,
    'competencia_tecnicas_e_comportamentais': competencias,
    'nivel_academico_y': nivel_academico,
    'nivel_ingles_y': nivel_ingles,
    'senioridade_y': senioridade,
    'area_atuacao_grupo': area_atuacao,
    'local_vaga': local_vaga
}


# Ação: Buscar candidatos
if st.button("🔍 Buscar Candidatos"):
    # Gerar variáveis de match
    df_match = gerar_variaveis_match(df_candidatos, vaga)

    #  Separar variáveis numéricas e matches
    X_numericas = df_match[[
        'match_competencias_v3',
        'match_titulo_vaga_perfil_v2',
        'match_local',
        'match_senioridade_aceitavel',
        'delta_ingles',
        'delta_academico',
        'delta_senioridade'
    ]]

    #  Selecionar colunas dummies (tipo e área)
    colunas_dummies = [col for col in df_match.columns if (
        col.startswith('tipo_contratacao_') or col.startswith('area_atuacao_grupo_')
    )]

    X_dummies = df_match[colunas_dummies]

    # Concatenar tudo
    X = pd.concat([X_numericas, X_dummies], axis=1)

    # Reindexar para garantir que as colunas estão iguais ao treinamento
    X = X.reindex(columns=colunas_modelo, fill_value=0)


    # Fazer predição
    proba = modelo.predict_proba(X)[:, 1]
    df_match['prob_contratacao'] = proba

    #  Ordenar pelos mais prováveis
    resultado = df_match.sort_values(by='prob_contratacao', ascending=False)


    #  Mostrar resultado
    st.subheader("✅ Candidatos recomendados:")
    st.dataframe(
        resultado[['codigo_candidato', 'titulo_profissional', 'prob_contratacao']].reset_index(drop=True)
    )

