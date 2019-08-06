import numpy as np
import gc
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


def build_en_emb(config, w2id):
    en_word2vec = KeyedVectors.load_word2vec_format(
        config["en_W2V_FILE"], binary=True)
    en_EMBEDDING_DIM = config["emb_dim"]
    # n_word<max_featureの時にerrになるよ
    vocabulary_size = min(config["en_voc"], config["max_features"])
    en_embedding_matrix = np.zeros((vocabulary_size, en_EMBEDDING_DIM))
    print("voc->", vocabulary_size)
    cnt = 0
    for word, i in w2id.items():
        if i == 0 or i == 1 or i == 2 or i == 3:
            continue
        try:
            en_embedding_vector = en_word2vec[word]
            en_embedding_matrix[i] = en_embedding_vector
        except KeyError:
            cnt += 1
            en_embedding_matrix[i] = np.random.normal(
                0, np.sqrt(0.25), en_EMBEDDING_DIM)
    print("UNK_rate", cnt/i)
    del en_word2vec
    gc.collect()
    return en_embedding_matrix


def build_jp_emb(config, w2id):
    jp_word2vec = Word2Vec.load(config["jp_W2V_FILE"])
    jp_EMBEDDING_DIM = config["emb_dim"]
    vocabulary_size = min(config["jp_voc"], config["max_features"])
    jp_embedding_matrix = np.zeros((vocabulary_size, jp_EMBEDDING_DIM))
    print("voc->", vocabulary_size)
    cnt = 0
    for word, i in w2id.items():
        if i == 0 or i == 1 or i == 2 or i == 3:
            continue
        try:
            jp_embedding_vector = jp_word2vec[word]
            jp_embedding_matrix[i] = jp_embedding_vector
        except KeyError:
            cnt += 1
            jp_embedding_matrix[i] = np.random.normal(
                0, np.sqrt(0.25), jp_EMBEDDING_DIM)
    print("UNK/rate->", cnt/i)

    del jp_word2vec
    gc.collect()
    return jp_embedding_matrix
