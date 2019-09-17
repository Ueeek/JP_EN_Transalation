import MeCab
from collections import Counter


class LangJa:
    def __init__(self, config):
        self.SOS_token = config["SOS_token"]
        self.EOS_token = config["EOS_token"]
        self.UNK_token = config["UNK_token"]
        self.max_features = config["jp_voc"]
        self.mask_token = config["mask_token"]
        self.word2index = {"SOS": self.SOS_token,
                           "EOS": self.EOS_token,
                           "UNK": self.UNK_token,
                           "MASK": self.mask_token
                           }
        self.index2word = {self.SOS_token: "SOS",
                           self.EOS_token: "EOS",
                           self.UNK_token: "UNK",
                           self.mask_token: "MASK"
                           }
        self.word2count = Counter()
        self.added = set()
        self.n_words = 4
        self.tagger = MeCab.Tagger("-Owakati")

    def add_from_df(self, df):
        for sentence in df:
            self._addSentence(sentence)
        self._register_word()

    def _addSentence(self, sentence):
        for word in self.tagger.parse(sentence).split():
            self._addWord(word)

    def _addWord(self, word):
        if word not in self.added:
            self.added.add(word)
            self.word2count.update([word])
        else:
            self.word2count[word] += 1
            pass

    def _register_word(self):
        for word, _ in self.word2count.most_common():
            if self.n_words >= self.max_features:
                break
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        print("register word to dict")

    def word2id(self, sentence, target=False):
        ret = []
        if target:
            ret.append(self.SOS_token)
        for word in self.tagger.parse(sentence).split():
            try:
                word_id = self.word2index[word]
            except KeyError:
                word_id = self.UNK_token
            ret.append(word_id)
        ret.append(self.EOS_token)
        return ret

    def id2word(self, ids):
        ret = []
        for id in ids:
            word = self.index2word[id]
            ret.append(word)
        return ret
