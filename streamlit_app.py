import streamlit as st
import pandas as pd
import joblib
from modelo import gerar_variaveis_match

# =======================
# Carregar modelo e dados
# =======================
modelo = joblib.load('modelo_xgb_final.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')  # Colunas do treino
df_candidatos = pd.read_csv('df_candidatos_tratado.csv')

# =======================
# Aba de Navegação
# =======================
aba = st.sidebar.selectbox(
    "## 📑 Selecione a Aba",
    ["Introdução e Metodologia", "Recomendação de Candidatos"]
)

# =======================
# Aba 1 - Introdução e Metodologia
# =======================
if aba == "Introdução e Metodologia":
    st.title("📑 Introdução e Metodologia")

    st.markdown("""
## 🔍 **Introdução**

Com o avanço da tecnologia e a crescente demanda por profissionais qualificados na área de Tecnologia da Informação (TI), os processos de recrutamento se tornam cada vez mais desafiadores. Encontrar o candidato ideal vai além de avaliar competências técnicas: é necessário considerar também critérios como aderência ao perfil da vaga, experiência, localização e senioridade.

Este trabalho propõe uma solução baseada em **Inteligência Artificial (IA)**, capaz de apoiar o processo de recrutamento da empresa **Decision**, especializada na alocação de profissionais de TI. O processo atual enfrenta desafios como falta de padronização nas avaliações, dificuldade em mensurar aderência dos candidatos e a necessidade de decisões rápidas que, muitas vezes, podem afetar a qualidade das contratações.

Por meio da aplicação de técnicas de **Machine Learning**, foi desenvolvido um modelo capaz de prever a probabilidade de um candidato ser contratado, a partir de dados históricos. A solução contempla também o desenvolvimento de uma **interface interativa em Streamlit**, que permite a simulação de vagas e a recomendação dos melhores candidatos.

O uso de IA no recrutamento permite aumentar a assertividade nas contratações, reduzir vieses e acelerar o processo seletivo.

---

## 🚀 **Metodologia**

### 1️⃣ Coleta e Entendimento dos Dados
Bases fornecidas pela Decision, contendo dados sobre candidatos, vagas e processos anteriores.

### 2️⃣ Análise Exploratória dos Dados (EDA)
Análise dos dados para identificação de inconsistências, dados ausentes e padrões relevantes.

### 3️⃣ Engenharia de Features
- **Match de Localização** (cidade e estado)
- **Match de Nível Acadêmico**
- **Match de Nível de Inglês**
- **Match de Senioridade**
- **Similaridade de Competências e Títulos** (PLN - Processamento de Linguagem Natural)

### 4️⃣ Pré-processamento
- Tratamento de valores nulos.
- One-Hot Encoding para variáveis categóricas.
- Agrupamento de áreas de atuação para criar variáveis robustas.

### 5️⃣ Modelagem
Modelos aplicados:
- **Regressão Logística** (baseline)
- **Random Forest**
- **XGBoost** (modelo final, melhor desempenho)

### 6️⃣ Desenvolvimento da Interface
Desenvolvimento de uma aplicação no Streamlit para simular vagas e recomendar candidatos.

---

## 🏁 **Conclusão**
O projeto demonstra como **Machine Learning** pode transformar o recrutamento, tornando-o mais eficiente, justo e escalável, contribuindo com a modernização do setor de Recursos Humanos.

---

## 📚 **Bibliografia**
- [ALBUQUERQUE, Ellen de Lima et al. - EnGeTec, 2022](https://www.fateczl.edu.br/engetec/engetec_2022/5_EnGeTec_paper_061.pdf)
- [BLUMEN, Daniel; CEPELLOS, Vanessa - Cadernos EBAPE.BR, 2023](https://www.scielo.br/j/cebape/a/5GNmGM3h3Yfrg96TX8mcTMC)
- [Huang, M. H. et al. - The Feeling Economy](https://www.researchgate.net/publication/348593461_The_Feeling_Economy_How_Artificial_Intelligence_Is_Creating_the_Era_of_Empathy)
- [TI Inside - Brasil aposta em IA, 2025](https://tiinside.com.br/12/03/2025/brasil-e-o-pais-que-mais-aposta-em-ai-revela-pesquisa-da-sap)
""")

# =======================
# Aba 2 - Recomendação
# =======================
if aba == "Recomendação de Candidatos":
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

    nivel_ingles = st.selectbox(
        "Nível de Inglês", 
        ["Básico", "Intermediário", "Avançado", "Fluente"]
    )

    local_vaga = st.text_input("Local da Vaga (Cidade, Estado) - Ex: São Paulo, São Paulo")


    # Filtros de Localização
    st.subheader("🎯 Filtros de Localização")
    filtro_local = st.checkbox('✅ Mostrar apenas candidatos da mesma **CIDADE**')
    filtro_estado = st.checkbox('✅ Mostrar apenas candidatos do mesmo **ESTADO**')

    # Montar dicionário da vaga
    vaga = {
        'titulo_vaga': titulo_vaga,
        'competencia_tecnicas_e_comportamentais': competencias,
        'nivel_academico_y': nivel_academico,
        'nivel_ingles_y': nivel_ingles,
        'senioridade_y': senioridade,
        'area_atuacao_grupo': area_atuacao,
        'local_vaga': local_vaga
    }

    # 🔍 Buscar candidatos
    if st.button("🔍 Buscar Candidatos"):
        # Gerar variáveis de match
        df_match = gerar_variaveis_match(df_candidatos, vaga)

        # Filtrar localização
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
            st.warning("⚠️ Verifique se o campo 'Local da vaga' foi preenchido corretamente no formato 'Cidade, Estado'.")

        # 🔸 Filtrar por Senioridade exata
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

        # Garantir alinhamento com o modelo
        X = X.reindex(columns=colunas_modelo, fill_value=0)

        # Predição
        proba = modelo.predict_proba(X)[:, 1]
        df_match['prob_contratacao'] = proba

        # Threshold fixo
        threshold = 0.4
        df_match['aprovado'] = (df_match['prob_contratacao'] >= threshold).astype(int)

        # Filtrar aprovados
        resultado = df_match[df_match['aprovado'] == 1].copy()
        resultado = resultado.sort_values(by='prob_contratacao', ascending=False)
        resultado = resultado.drop_duplicates(subset='codigo_candidato')

        st.subheader("✅ Candidatos recomendados:")
        st.dataframe(
            resultado[['codigo_candidato', 'titulo_profissional', 'conhecimentos_tecnicos', 'local', 'prob_contratacao']].reset_index(drop=True)
        )
