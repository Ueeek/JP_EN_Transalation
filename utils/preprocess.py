import MeCab
import pandas as pd
import unicodedata
import string
import re
#import random


def _unicodeToAscii(s):
    return " ".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn")


def _norm_en(s):
    ret = []
    for c in s.split():
        c = _unicodeToAscii(c.lower())
        c = re.sub(r"([.!?])", r" \1", c)
        c = re.sub(r"[^a-zA-Z.!?]+", r" ", c)
        ret.append("".join(c.split()))
    return " ".join(ret)


def _norm_jp(sentence):
    tagger = MeCab.Tagger("-Owakati")
    ret = []
    for word in tagger.parse(sentence).split():
        word = re.sub(r"([、。])", r"\1", word)
        ret.append("".join(word.split()))
    return " ".join(ret)


def _length_filter(data, config):
    return data[data.apply(lambda x:len(x[config["en_col"]].split(" ")), axis=1) < config["MAX_LENGTH"]]


def loadLangs(config):
    en_col = config["en_col"]
    jp_col = config["jp_col"]
    print("reading lines")
    data = pd.read_csv(config["corpus_file"], sep="\t", names=[en_col, jp_col])
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    data[en_col] = data[en_col].apply(lambda x: "".join(_norm_en(x)))
    data[jp_col] = data[jp_col].apply(lambda x: "".join(_norm_jp(x)))
    data = data[[en_col, jp_col]]
    filtered = _length_filter(data, config)
    return filtered
