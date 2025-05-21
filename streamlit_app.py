import streamlit as st
import pandas as pd
import joblib
from modelo import gerar_variaveis_match

# Carregar modelo e dados
modelo = joblib.load('modelo_xgb_final.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')  # As colunas usadas no treino
df_candidatos = pd.read_csv('df_candidatos_tratado.csv')

st.title("🔍 Recomendação de Candidatos para Vaga")

# 🔸 Inputs da vaga
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
        'Doutorado Cursando', 'Doutorado Incompleto', 'Doutorado Completo'
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


# 🔸 Filtros Geográficos
st.subheader("🎯 Filtros de Localização")

filtro_local = st.checkbox('✅ Mostrar apenas candidatos da mesma **CIDADE**')
filtro_estado = st.checkbox('✅ Mostrar apenas candidatos do mesmo **ESTADO**')


# 🔸 Montar dicionário da vaga
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
    # 🔸 Gerar variáveis de match
    df_match = gerar_variaveis_match(df_candidatos, vaga)

    # 🔸 Aplicar filtros de localização
    # 🔸 Separar cidade e estado da vaga
    try:
        cidade_vaga = vaga['local_vaga'].split(",")[0].strip().lower()
        estado_vaga = vaga['local_vaga'].split(",")[1].strip().lower()

    # Criar colunas auxiliares para cidade e estado dos candidatos
        df_match[['cidade_candidato', 'estado_candidato']] = df_match['local'].str.split(",", n=1, expand=True)
        df_match['cidade_candidato'] = df_match['cidade_candidato'].str.strip().str.lower()
        df_match['estado_candidato'] = df_match['estado_candidato'].str.strip().str.lower()

    # 🔸 Aplicar filtro de cidade
        if filtro_local:
            df_match = df_match[df_match['cidade_candidato'] == cidade_vaga]

    # 🔸 Aplicar filtro de estado
        if filtro_estado:
            df_match = df_match[df_match['estado_candidato'] == estado_vaga]

    except Exception:
        st.warning("⚠️ Verifique se o campo 'Local da vaga' foi preenchido corretamente no formato 'Cidade, Estado'.")


    

    # 🔸 Selecionar as variáveis do modelo
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

    # 🔥 Garantir que as colunas estão alinhadas com o modelo
    X = X.reindex(columns=colunas_modelo, fill_value=0)

    # 🔸 Fazer predição
    proba = modelo.predict_proba(X)[:, 1]
    df_match['prob_contratacao'] = proba

    # 🔥 Threshold fixo
    threshold = 0.4

    # Aplicar threshold
    df_match['aprovado'] = (df_match['prob_contratacao'] >= threshold).astype(int)

    # 🔸 Filtrar os candidatos aprovados
    resultado = df_match[df_match['aprovado'] == 1].copy()

    # 🔸 Ordenar por maior probabilidade
    resultado = resultado.sort_values(by='prob_contratacao', ascending=False)

    # 🔸 Remover candidatos duplicados (caso tenha)
    resultado = resultado.drop_duplicates(subset='codigo_candidato')

    # 🔸 Mostrar resultado
    st.subheader("✅ Candidatos recomendados:")
    st.dataframe(
        resultado[['codigo_candidato', 'titulo_profissional', 'conhecimentos_tecnicos', 'local', 'prob_contratacao']].reset_index(drop=True)
    )
