import streamlit as st
import pandas as pd
import joblib
from modelo import gerar_variaveis_match

# Configurações da página
st.set_page_config(page_title="Recomendação de Candidatos")

# =======================
# Carregar modelo e dados
# =======================
modelo = joblib.load('modelo_xgb_final.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')
df_candidatos = pd.read_csv('df_candidatos_tratado.csv')

# =======================
# Cabeçalho com imagem
# =======================
st.title("🔍 Recomendação Inteligente de Candidatos com IA")

# =======================
# Criação das Abas
# =======================
tab1, tab2 = st.tabs(["📑 Introdução e Metodologia", "🎯 Recomendação de Candidatos"])

# =======================
# Aba 1 - Introdução
# =======================
with tab1:
    st.header("📑 Introdução e Metodologia")
    st.markdown("""
### 🔍 **Introdução**

Com o avanço da tecnologia e a crescente demanda por profissionais qualificados na área de Tecnologia da Informação (TI), os processos de recrutamento se tornam cada vez mais desafiadores. Encontrar o candidato ideal vai além de avaliar competências técnicas: é necessário considerar também critérios como aderência ao perfil da vaga, experiência, localização e senioridade.

Este trabalho propõe uma solução baseada em **Inteligência Artificial (IA)**, capaz de apoiar o processo de recrutamento da empresa **Decision**, especializada na alocação de profissionais de TI.

Por meio da aplicação de técnicas de **Machine Learning**, foi desenvolvido um modelo capaz de prever a probabilidade de um candidato ser contratado, a partir de dados históricos. A solução contempla também uma **interface interativa em Streamlit**, que permite a simulação de vagas e a recomendação dos melhores candidatos.

---

### 🚀 **Metodologia**

1️⃣ **Coleta e Entendimento dos Dados**  
Bases fornecidas pela Decision, contendo dados sobre candidatos, vagas e processos anteriores.

2️⃣ **Análise Exploratória dos Dados (EDA)**  
Análise dos dados para identificação de inconsistências, dados ausentes e padrões relevantes.

3️⃣ **Engenharia de Features**  
- Match de Localização (cidade e estado)  
- Match de Nível Acadêmico  
- Match de Nível de Inglês  
- Match de Senioridade  
- Similaridade de Competências e Títulos (PLN)

4️⃣ **Pré-processamento**  
- Tratamento de valores nulos  
- One-Hot Encoding para variáveis categóricas  
- Agrupamento de áreas de atuação para criar variáveis robustas  

5️⃣ **Modelagem**  
Modelos aplicados:  
- Regressão Logística (baseline)  
- Random Forest  
- XGBoost (modelo final, melhor desempenho)  

6️⃣ **Desenvolvimento da Interface**  
Aplicação no Streamlit para simular vagas e recomendar candidatos.

---

### 🏁 **Conclusão**  

O desenvolvimento deste projeto demonstra como a aplicação de técnicas de Machine Learning pode transformar processos de recrutamento, tornando-os mais eficientes, precisos e escaláveis. 
Durante os testes, o uso de um threshold de 0.3 apresentou um comportamento alinhado ao objetivo do RH: maximizar o recall, ou seja, garantir que o maior número possível de bons candidatos não fosse descartado. Com esse limiar, o modelo priorizou ser mais sensível na identificação de perfis com aderência às vagas, ainda que, como contrapartida, haja um aumento na taxa de falsos positivos. Essa decisão é coerente com processos seletivos, onde um volume um pouco maior de perfis pode ser analisado manualmente, desde que os melhores talentos não sejam perdidos.
O modelo final, utilizando o XGBoost, apresentou resultados aceitáveis, sobretudo nas métricas de recall e F1-score, quando comparado aos modelos base de Regressão Logística e Random Forest. Além disso, o desenvolvimento da interface em Streamlit trouxe uma camada de acessibilidade e interatividade que permite aos recrutadores simular diferentes cenários de vagas, filtrando por critérios como localização (cidade/estado), senioridade e outros fatores relevantes.

---

### ⭐ **Resultado Prático**
A solução desenvolvida já permite à Decision reduzir tempo de triagem, aumentar assertividade na pré-seleção e garantir mais padronização no processo de matching entre vagas e candidatos.

Ao final, a proposta demonstra a viabilidade técnica, além de abrir portas para uma jornada de transformação digital no setor de Recursos Humanos.


---
### 📚 **Referências**  
- [Huang, M. H. et al. - The Feeling Economy](https://www.researchgate.net/publication/348593461_The_Feeling_Economy_How_Artificial_Intelligence_Is_Creating_the_Era_of_Empathy)  
- [TI Inside - Brasil aposta em IA, 2025](https://tiinside.com.br/12/03/2025/brasil-e-o-pais-que-mais-aposta-em-ai-revela-pesquisa-da-sap)  
    """)

# =======================
# Aba 2 - Recomendação
# =======================
with tab2:
    st.header("🎯 Recomendação de Candidatos para Vaga")

    # Dados da vaga
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


    # Filtros
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

        # 🔸 Filtrar localização
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

        except:
            st.warning("⚠️ Verifique se o campo 'Local da vaga' foi preenchido corretamente no formato 'Cidade, Estado'.")

        # 🔸 Filtrar por senioridade exata
        df_match = df_match[df_match['nivel_profissional'].str.lower() == vaga['senioridade_y'].lower()]

        # 🔸 Variáveis do modelo
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

        # 🔥 Threshold fixo
        threshold = 0.4
        df_match['aprovado'] = (df_match['prob_contratacao'] >= threshold).astype(int)

        resultado = df_match[df_match['aprovado'] == 1].copy()
        resultado = resultado.sort_values(by='prob_contratacao', ascending=False)
        resultado = resultado.drop_duplicates(subset='codigo_candidato')

        st.subheader("✅ Candidatos recomendados:")
        st.dataframe(
            resultado[['codigo_candidato', 'titulo_profissional', 'conhecimentos_tecnicos', 'local', 'prob_contratacao']].reset_index(drop=True)
        )
