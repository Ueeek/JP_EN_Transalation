# Result
## S2S_simple.py
### model
* encoder: bidirectional GRU * 2 layer
* decoder: bidirectional GRU * 2 layer
### performance
* 翻訳は成功した。(20epochくらいで、train_lossはいい感じに収束したっぽい?val lossは収束する気がなさそう)
* overfittingが激しい
* trainデータの翻訳は、いい感じにできるけど、valデータはまーまーといったところ

## S2S_normalize.py
### model
* S2S_simple+
    * batch_norm layer
    * gru + dropout

### performance
* あまり性能に変化がなかった。
* drop rateとかを変えてみると変わるのかな

## S2S_emb.py
### model
* S2S_+ pretrained_embedding

### performance
* 
