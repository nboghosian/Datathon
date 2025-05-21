import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Limpeza de texto
def limpar_texto(texto):
    if pd.isna(texto):
        return ''
    return texto.lower().replace('\n', ' ').replace('.', '').strip()

# Similaridade TF-IDF entre dois textos
def calcular_similaridade_tfidf(texto1, texto2):
    if not texto1 or not texto2:
        return 0.0
    corpus = [texto1, texto2]
    vectorizer = TfidfVectorizer().fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity, 3)

# Match de localização
def calcular_match_local(vaga_local, cand_local):
    if pd.isna(vaga_local) or pd.isna(cand_local):
        return 0
    return int(vaga_local.strip().lower() == cand_local.strip().lower())

# Match de nível acadêmico
def calcular_delta_academico(vaga, candidato):
    mapa = {
        "ensino médio": 1,
        "superior incompleto": 2,
        "superior completo": 3,
        "pós-graduação": 4,
        "mestrado": 5,
        "doutorado": 6
    }
    v = mapa.get(vaga.lower(), 0)
    c = mapa.get(candidato.lower(), 0)
    return c - v

def calcular_match_nivel_academico(vaga, candidato):
    delta = calcular_delta_academico(vaga, candidato)
    return int(delta >= 0)

# Match de nível de inglês (mesma lógica do acadêmico)
def calcular_delta_ingles(vaga, candidato):
    mapa = {
        "básico": 1,
        "intermediário": 2,
        "avançado": 3,
        "fluente": 4
    }
    v = mapa.get(vaga.lower(), 0)
    c = mapa.get(candidato.lower(), 0)
    return c - v

def calcular_match_ingles(vaga, candidato):
    delta = calcular_delta_ingles(vaga, candidato)
    return int(delta >= 0)

# Função principal que processa tudo
def gerar_variaveis_match(df_candidatos, vaga):
 
    # Padronizar nomes de colunas (resolver problemas de _x e _y)
    df = df_candidatos.rename(columns={
        'nivel_ingles_x': 'nivel_ingles',
        'nivel_academico_x': 'nivel_academico'
    }).copy()
    

    # Texto para perfil
    df['perfil_candidato'] = df['titulo_profissional'].fillna('') + ' ' + df['objetivo_profissional'].fillna('')

    # Matches de texto
    df['match_competencias'] = df.apply(lambda x: calcular_similaridade_tfidf(
        limpar_texto(vaga['competencia_tecnicas_e_comportamentais']),
        limpar_texto(x['conhecimentos_tecnicos'])
    ), axis=1)

    df['match_titulo_vaga_perfil'] = df.apply(lambda x: calcular_similaridade_tfidf(
        limpar_texto(vaga['titulo_vaga']),
        limpar_texto(x['perfil_candidato'])
    ), axis=1)

    # Matches de localização
    df['match_local'] = df.apply(lambda x: calcular_match_local(
        vaga['local_vaga'], x['local']
    ), axis=1)

    # Matches de inglês e acadêmico
    df['delta_ingles'] = df.apply(lambda x: calcular_delta_ingles(
        vaga['nivel_ingles_y'], x['nivel_ingles']
    ), axis=1)

    df['match_ingles'] = (df['delta_ingles'] >= 0).astype(int)

    df['delta_academico'] = df.apply(lambda x: calcular_delta_academico(
        vaga['nivel_academico_y'], x['nivel_academico']
    ), axis=1)

    df['match_nivel_academico'] = (df['delta_academico'] >= 0).astype(int)

    return df
