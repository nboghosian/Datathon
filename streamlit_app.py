import streamlit as st
import pandas as pd
import joblib
from modelo import gerar_variaveis_match

# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="Recomendação de Candidatos", layout="wide")

# =========================
# CSS para tema escuro nas abas
# =========================
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
        color: white;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #000000;
        color: white;
        border-radius: 5px;
        padding: 8px;
        margin-right: 4px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333333;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Cabeçalho com Imagem
# =========================
st.image("recrutamento.jpg", use_column_width=True)

# =========================
# Carregar modelo e dados
# =========================
modelo = joblib.load('modelo_xgb_final.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')
df_candidatos = pd.read_csv('df_candidatos_tratado.csv')

# =========================
# Criação das Abas
# =========================
abas = st.tabs(["📑 Introdução e Metodologia", "🔍 Recomendação de Candidatos"])

# =========================
# 📑 Aba 1 - Introdução
# =========================
with abas[0]:
    st.markdown("""
    # 📑 Introdução e Metodologia

    ## 🔍 Introdução

    Este projeto foi desenvolvido para otimizar o processo de recrutamento na **Decision**, utilizando **Machine Learning** e uma aplicação interativa em **Streamlit**.

    O objetivo é fornecer recomendações assertivas para o time de recrutamento, levando em conta:
    - Competências técnicas
    - Similaridade de título profissional
    - Aderência em localização (cidade/estado)
    - Formação acadêmica
    - Senioridade
    - Nível de inglês

    ## 🚀 Metodologia

    1️⃣ Coleta e tratamento dos dados históricos da Decision  
    2️⃣ Engenharia de features de "match" (local, acadêmico, inglês, título, competências, senioridade)  
    3️⃣ Processamento de linguagem natural (TF-IDF) para comparar textos  
    4️⃣ Modelagem com **XGBoost**, além de Random Forest e Regressão Logística para comparação  
    5️⃣ Deploy em aplicativo web interativo usando **Streamlit**

    ## 🏁 Conclusão

    A aplicação permite uma triagem mais ágil, precisa e objetiva, auxiliando o RH na tomada de decisão baseada em dados.
    """)

# =========================
# 🔍 Aba 2 - Recomendação
# =========================
with abas[1]:
    st.title("🔍 Recomendação de Candidatos para Vaga")

    # Inputs da vaga
    st.subheader("📄 Dados da Vaga")

    titulo_vaga = st.text_input("Título da Vaga")

    senioridade = st.selectbox(
        "Nível", 
        ["Estagiário", "Auxiliar", "Assistente", "Júnior", "Pleno", "Sênior", 
         "Especialista", "Coordenador", "Gerente", "Supervisor"]
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
            'Ensino Superior Completo','Ensino Superior Incompleto', 'Ensino Superior Cursando',
            'Pós Graduação Incompleto', 'Pós Graduação Cursando', 'Pós Graduação Completo',
            'Mestrado Cursando', 'Mestrado Incompleto', 'Mestrado Completo',
            'Doutorado Cursando', 'Doutorado Incompleto', 'Doutorado Completo',
            'Ensino Médio Incompleto', 'Ensino Médio Cursando', 'Ensino Médio Completo',
            'Ensino Técnico Incompleto', 'Ensino Técnico Cursando', 'Ensino Técnico Completo',
            'Ensino Fundamental Incompleto', 'Ensino Fundamental Cursando', 'Ensino Fundamental Completo'
        ]
    )

    nivel_ingles = st.selectbox("Nível de Inglês", ["Básico", "Intermediário", "Avançado", "Fluente"])

    local_vaga = st.text_input("Local da Vaga (Cidade, Estado) - Ex: São Paulo, São Paulo")

    # Filtros de localização
    st.subheader("🎯 Filtros de Localização")
    filtro_local = st.checkbox('✅ Mostrar apenas candidatos da mesma **CIDADE**')
    filtro_estado = st.checkbox('✅ Mostrar apenas candidatos do mesmo **ESTADO**')

    vaga = {
        'titulo_vaga': titulo_vaga,
        'competencia_tecnicas_e_comportamentais': competencias,
        'nivel_academico_y': nivel_academico,
        'nivel_ingles_y': nivel_ingles,
        'senioridade_y': senioridade,
        'area_atuacao_grupo': area_atuacao,
        'local_vaga': local_vaga
    }

    if st.button("🔍 Buscar Candidatos"):
        df_match = gerar_variaveis_match(df_candidatos, vaga)

        try:
            cidade_vaga = vaga['local_vaga'].split(",")[0].strip().lower()
            estado_vaga = vaga['local_vaga'].split(",")[1].strip().lower()

            df_match[['cidade_candidato', 'estado_candidato']] = df_match['local'].str.split(",", n=1, expand=True)
            df_match['cidade_candidato'] = df_match['cidade_candidato'].str.strip().str.lower()
            df_match['estado_candidato'] = df_match['estado_candidato'].str.strip().str.lower()

            if filtro_local:
                df_match = df_match[df_match['cidade_candidato'] == cidade_vaga]
            if filtro_estado:
                df_match = df_match[df_match['estado_candidato'] == estado_vaga]

        except Exception:
            st.warning("⚠️ Verifique se o campo 'Local da vaga' está no formato correto: 'Cidade, Estado'.")

        # 🔥 Filtro obrigatório por senioridade exata
        df_match = df_match[df_match['nivel_profissional'].str.lower() == vaga['senioridade_y'].lower()]

        # Variáveis do modelo
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

        X = X.reindex(columns=colunas_modelo, fill_value=0)

        proba = modelo.predict_proba(X)[:, 1]
        df_match['prob_contratacao'] = proba

        threshold = 0.4
        df_match['aprovado'] = (df_match['prob_contratacao'] >= threshold).astype(int)

        resultado = df_match[df_match['aprovado'] == 1].copy()
        resultado = resultado.sort_values(by='prob_contratacao', ascending=False)
        resultado = resultado.drop_duplicates(subset='codigo_candidato')

        st.subheader("✅ Candidatos recomendados:")
        st.dataframe(
            resultado[['codigo_candidato', 'titulo_profissional', 'conhecimentos_tecnicos', 'local', 'prob_contratacao']].reset_index(drop=True)
        )
