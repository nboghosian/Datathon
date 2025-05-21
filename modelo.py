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
        'ensino fundamental incompleto': 0,
        'ensino fundamental cursando': 1,
        'ensino fundamental completo': 2,
        'ensino médio incompleto': 3,
        'ensino médio cursando': 4,
        'ensino médio completo': 5,
        'ensino técnico incompleto': 6,
        'ensino técnico cursando': 7,
        'ensino técnico completo': 8,
        'ensino superior incompleto': 9,
        'ensino superior cursando': 10,
        'ensino superior completo': 11,
        'pós graduacao incompleto': 12,
        'pós graduacao cursando': 13,
        'pós graduacao completo': 14,
        'mestrado cursando': 15,
        'mestrado incompleto': 16,
        'mestrado completo': 17,
        'doutorado cursando': 18,
        'doutorado incompleto': 19,
        'doutorado completo': 20,
        'não informado': -1,
        '': -1
    }
    v = mapa.get(str(vaga).strip().lower(), 0)
    c = mapa.get(str(candidato).strip().lower(), 0)
    return c - v

def calcular_delta_senioridade(vaga, candidato):
    mapa = {
        'aprendiz': 0,
        'auxiliar': 1,
        'estagiário': 2,
        'trainee': 3,
        'assistente': 4,
        'técnico de nível médio': 5,
        'júnior': 6,
        'analista': 7,
        'pleno': 8,
        'especialista': 9,
        'líder': 10,
        'supervisor': 11,
        'coordenador': 12,
        'sênior': 13,
        'gerente': 14,
        'não informado': -1,
        '': -1
    }
    v = mapa.get(str(vaga).strip().lower(), 0)
    c = mapa.get(str(candidato).strip().lower(), 0)
    return c - v

def calcular_match_senioridade(delta):
    return int(delta >= 0)  # 1 se o candidato tem a senioridade igual ou superior

def calcular_match_nivel_academico(vaga, candidato):
    delta = calcular_delta_academico(vaga, candidato)
    return int(delta >= 0)

# Match de nível de inglês (mesma lógica do acadêmico)
def calcular_delta_ingles(vaga, candidato):
    mapa = {
        'nenhum': 0,
        'básico': 1,
        'intermediário': 2,
        'técnico': 3,
        'avançado':4,
        'fluente': 5,
        'não informado': -1,
        '': -1
    }
    v = mapa.get(str(vaga).strip().lower(), 0)
    c = mapa.get(str(candidato).strip().lower(), 0)
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
    
    # Match de área de atuação
    df['match_area_atuacao'] = (df['area_atuacao_grupo'] == vaga['area_atuacao_grupo']).astype(int)

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

    df['delta_senioridade'] = df.apply(lambda x: calcular_delta_senioridade(
    vaga.get('nivel_profissional_y', 'nao informado'),
    x.get('nivel_profissional', 'nao informado')
    ), axis=1)

    df['match_senioridade_aceitavel'] = df['delta_senioridade'].apply(lambda x: 1 if x <= 1 else 0)


    return df
