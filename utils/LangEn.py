import mosestokenizer as mos


class LangEn:
    """
    register Engslish word
    assign id to each word
    """

    def __init__(self, config):
        self.SOS_token = config["SOS_token"]
        self.EOS_token = config["EOS_token"]
        self.UNK_token = config["UNK_token"]
        self.max_features = config["en_voc"]

        self.word2index = {"SOS": self.SOS_token,
                           "EOS": self.EOS_token, "UNK": self.UNK_token}
        self.word2count = {}
        self.index2word = {self.SOS_token: "SOS",
                           self.EOS_token: "EOS", self.UNK_token: "UNK"}
        self.n_words = 4
        self.tokenizer = mos.MosesTokenizer("en")

    def addSentence(self, sentence):
        for word in self.tokenizer(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            if self.n_words >= self.max_features:
                return
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def word2id(self, sentence, target=False):
        ret = []
        if target:
            ret.append(self.SOS_token)
        for word in self.tokenizer(sentence):
            try:
                word_id = self.word2index[word]
            except KeyError:
                word_id = self.word2index["UNK"]
            ret.append(word_id)
        ret.append(self.EOS_token)
        return ret

    def id2word(self, ids):
        ret = []
        for id in ids:
            try:
                word = self.index2word[id]
            except KeyError:
                word = "UNK"
            ret.append(word)
        return ret
