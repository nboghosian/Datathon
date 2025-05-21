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
    # 🔸 Gerar variáveis de match
    df_match = gerar_variaveis_match(df_candidatos, vaga)

    # 🔸 Selecionar variáveis usadas no treino
    X = df_match[[
        'area_atuacao_grupo_desenvolvimento',
        'area_atuacao_grupo_governanca',
        'area_atuacao_grupo_infraestrutura',
        'area_atuacao_grupo_nao_informado',
        'area_atuacao_grupo_negocio/adm',
        'area_atuacao_grupo_outros',
        'area_atuacao_grupo_projetos',
        'area_atuacao_grupo_qualidade',
        'area_atuacao_grupo_sap',
        'area_atuacao_grupo_seguranca',
        'area_atuacao_grupo_ux',
        'delta_academico',
        'delta_ingles',
        'delta_senioridade',
        'match_senioridade_aceitavel',
        'match_competencias_v3',
        'match_titulo_vaga_perfil_v2',
        'match_local'
    ]]

    # 🔥 Garantir alinhamento com o modelo treinado
    X = X.reindex(columns=colunas_modelo, fill_value=0)

    # ==========================
    # 🔸 Fazer predição
    # ==========================
    proba = modelo.predict_proba(X)[:, 1]
    df_match['prob_contratacao'] = proba

    # Definir o threshold ideal
    threshold = 0.3  # Você pode deixar fixo ou criar um st.slider para ajustar

    # Aplicar threshold manual
    df_match['aprovado'] = (df_match['prob_contratacao'] >= threshold).astype(int)

    # Filtrar os candidatos aprovados
    resultado = df_match[df_match['aprovado'] == 1].copy()

    # Ordenar por maior probabilidade
    resultado = resultado.sort_values(by='prob_contratacao', ascending=False)

    # Remover candidatos duplicados (caso tenha)
    resultado = resultado.drop_duplicates(subset='codigo_candidato')

    # ==========================
    # 🔸 Mostrar resultado
    # ==========================
    st.subheader("✅ Candidatos recomendados:")
    st.dataframe(
        resultado[['codigo_candidato', 'titulo_profissional','conhecimentos_tecnicos','local', 'prob_contratacao']].reset_index(drop=True)
    )
