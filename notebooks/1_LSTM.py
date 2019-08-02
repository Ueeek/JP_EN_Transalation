#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import gc
import os
sys.path.append("/home/ueki.k/jp_en_translation")


# In[4]:


from models.Seq2Seq_1 import build_model
from utils.LangEn import LangEn
from utils.LangJa import LangJa
from utils.preprocess import loadLangs
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


# In[ ]:


config={\
    "corpus_file":"../data/jpn.txt",\
    "en_col":"description_en",\
    "jp_col":"description_jp",\
    "SOS_token":1,\
    "EOS_token":0,\
    "UNK_token":2,\
    "max_features":5000,\
    "MAX_LENGTH":20,\
    "train_size":15000,\
    "val_size":100,\
    "batch_size":128,\
    "epochs":111,\
    "maxlen_enc":20,\
    "maxlen_dec":20,\
    "n_hidden":400,\
    "input_dim":5000,\
    "output_dim":5000,\
    "emb_dim":300,\
    "use_enc_emb":False,\
    "use_dec_emb":False,\
    "validation_split":0.1,\
    "trained_param_dir":"../trained_models/1_lstm_ja_en_00.hdf5",\
    "translate_length":25,\
    "en_W2V_FILE" : "../data/GoogleNews-vectors-negative300.bin.gz",\
    "jp_W2V_FILE":"../data/ja_data/ja.bin",\
    "src":"en",\
    "trg":"jp",\
}


# In[ ]:


class Trainer:
    def __init__(self,config):
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.validation_split = config["validation_split"]
        self.trained_param_dir = config["trained_param_dir"]
        self.output_dim = config["output_dim"]
        self.hist =None
    def train(self,e_input,d_input,target):
        print("#1 train procedure start")
        model,_,_ = build_model(config)
        model.summary()
        
        if os.path.isfile(self.trained_param_dir): #モデルの学習済みパラメータ
            print("2-1? load param")
            model.load_weights(self.trained_param_dir)
        else:
            print("no_emb")
        print("#6 start training")
        
        target_categorical = np_utils.to_categorical(output_target_padded,self.output_dim)
       
        self.hist=model.fit([e_input,d_input],target_categorical,epochs=self.epochs,batch_size=self.batch_size,validation_split=self.validation_split)
        print("#9 save_param")
        model.save_weights(self.trained_param_dir)
        #return model    


# In[ ]:


class Translator:
    def __init__(self,config):
        self.translate_length = config["translate_length"]
        self.trained_param_dir = config["trained_param_dir"]
        self.model,self.encoder,self.decoder = build_model(config,test=True)
        self.model.load_weights(self.trained_param_dir)
    ## 翻訳文生成
    def _translate(self,e_input):
        #encode input to vec
        #encoder_outputs,state_h_1,state_c_1 = self.encoder.predict(e_input)
        #states_values=[state_h_1,state_c_1]
        encoder_outputs,*states_values = self.encoder.predict(e_input)
        
        #first token
        target_seq=np.zeros((1,1))
        target_seq[0,0] = config["SOS_token"]
        
        decoded_sentence=[]
        for i in range(0,self.translate_length):
            #output_tokens,h1,c1 = self.decoder.predict([target_seq]+states_values)
            output_tokens,*states_values = self.decoder.predict([target_seq]+states_values)
            
            sampled_token_index=np.argmax(output_tokens[0,0,:])
            if sampled_token_index==config["EOS_token"]:
                decoded_sentence.append(config["EOS_token"])
                break
            else:
                target_seq[0,0] = sampled_token_index
                #states_values =[h1,c1]
                decoded_sentence.append(sampled_token_index)
        return decoded_sentence                                    
    
    
    
    def translate_demo(self,src_data_id_seq):
        ret=[]
        for src in src_data_id_seq:
            id_seq_mat = np.array([src])
            pred_id_padded = sequence.pad_sequences(id_seq_mat,maxlen=config["MAX_LENGTH"],padding="post",truncating="post")
            pred=self._translate(pred_id_padded)
            ret.append(pred)
        return ret


# In[ ]:


def build_en_emb(config):
    en_word2vec= KeyedVectors.load_word2vec_format(config["en_W2V_FILE"],binary=True)
    en_EMBEDDING_DIM=config["emb_dim"]
    #n_word<max_featureの時にerrになるよ
    vocabulary_size=min(EN_lang.n_words,config["max_features"])
    en_embedding_matrix = np.zeros((vocabulary_size, en_EMBEDDING_DIM))
    print("voc->",vocabulary_size)
    cnt=0
    for word, i in EN_lang.word2index.items():
        if   i==0 or i==1 or i ==2:
            continue
        try:
            en_embedding_vector = en_word2vec[word]
            en_embedding_matrix[i] = en_embedding_vector
        except KeyError:
            cnt+=1
            en_embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),en_EMBEDDING_DIM)
    print("UNK_rate",cnt/i)
    del en_word2vec
    gc.collect()
    return en_embedding_matrix


# In[ ]:


def build_jp_emb(config):
    jp_word2vec= model = Word2Vec.load(config["jp_W2V_FILE"])
    jp_EMBEDDING_DIM=config["emb_dim"]
    vocabulary_size=min(JP_lang.n_words,config["max_features"])
    jp_embedding_matrix = np.zeros((vocabulary_size, jp_EMBEDDING_DIM))
    print("voc->",vocabulary_size)
    cnt=0
    for word, i in JP_lang.word2index.items():
        if   i==0 or i==1 or i ==2:
            continue
        try:
            jp_embedding_vector = jp_word2vec[word]
            jp_embedding_matrix[i] = jp_embedding_vector
        except KeyError:
            cnt+=1
            jp_embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),jp_EMBEDDING_DIM)
    print("UNK/rate->",cnt/i)

    del jp_word2vec
    gc.collect()
    return jp_embedding_matrix


# # train

# In[ ]:


data=loadLangs(config)


# In[ ]:


val_data = data[config["train_size"]:config["train_size"]+config["val_size"]]
data = data[:config["train_size"]]


# In[ ]:


EN_lang = LangEn(config)
JP_lang = LangJa(config)


# In[ ]:


for s in data[config["en_col"]]:
    EN_lang.addSentence(s)


# In[ ]:


for s in data[config["jp_col"]]:
    JP_lang.addSentence(s)


# ## input の加工

# In[ ]:


if config["src"]=="jp":
    src_col=config["jp_col"]
    trg_col=config["en_col"]
    Langs={"src":JP_lang,"trg":EN_lang}
else:
    src_col=config["en_col"]
    trg_col=config["jp_col"]
    Langs={"trg":JP_lang,"src":EN_lang}


# In[ ]:


input_en = data[src_col]


# In[ ]:


input_source_lang=data[src_col].apply(lambda x:Langs["src"].word2id(x))
input_target_lang=data[trg_col].apply(lambda x:Langs["trg"].word2id(x,target=True))
output_target_lang=data[trg_col].apply(lambda x:Langs["trg"].word2id(x))


# In[ ]:


input_source_padded=sequence.pad_sequences(input_source_lang,maxlen=config["MAX_LENGTH"],padding="post",truncating="post")
input_target_padded=sequence.pad_sequences(input_target_lang,maxlen=config["MAX_LENGTH"],padding="post",truncating="post")
output_target_padded=sequence.pad_sequences(output_target_lang,maxlen=config["MAX_LENGTH"],padding="post",truncating="post")


# In[ ]:


trainer = Trainer(config)


# In[ ]:


trainer.train(input_source_padded,input_target_padded,output_target_padded)


# In[ ]:


del trainer
gc.collect()


# # test

# In[ ]:


val_data_id = val_data[src_col].apply(lambda x:Langs["src"].word2id(x))


# In[ ]:


val_data_id[:10]


# In[ ]:


translator = Translator(config)


# In[ ]:


ret = translator.translate_demo(val_data_id)


# In[ ]:


for src,pred,target in zip(val_data[src_col],ret,val_data[trg_col]):
    print("src->",src)
    print()
    print("pred->"," ".join(Langs["trg"].id2word(pred)))
    print("ans->",target)
    print("------------------")


# In[ ]:




