from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.layers import LSTM, Embedding, BatchNormalization
from keras.initializers import uniform

def build_model(config, enc_emb_mat=None, dec_emb_mat=None,test=False):
    maxlen_e = config["maxlen_enc"]
    maxlen_d = config["maxlen_dec"] if not test else 1
    n_hidden = config["n_hidden"]
    input_dim = config["input_dim"]
    emb_dim = config["emb_dim"]
    output_dim = config["output_dim"]
    use_enc_emb = config["use_enc_emb"]
    use_dec_emb = config["use_dec_emb"]

    # building model
    print("#3 encoder")
    # encoder
    # input
    encoder_input = Input(shape=(maxlen_e,), name="encoder_input")
    # embedding
    if use_enc_emb and not test:
        emb_input = Embedding(output_dim=emb_dim, input_dim=input_dim, weights=[
            enc_emb_mat])(encoder_input)

    else:
        emb_input = Embedding(output_dim=emb_dim, input_dim=input_dim,
                              embeddings_initializer=uniform())(encoder_input)

    # batch_norm
    emb_input = BatchNormalization(axis=-1)(emb_input)

    # LSMT(forwarding)
    enc_fw1, state_h_fw1, state_c_fw1 = LSTM(n_hidden,
                                             name="encoder_LSTM_fw1",
                                             return_sequences=True,
                                             return_state=True)(emb_input)

    # LSTM(backword)
    enc_bw1, state_h_bw1, state_c_bw1 = LSTM(n_hidden,
                                             name="encoder_LSTM_bw1",
                                             return_sequences=True,
                                             return_state=True)(emb_input)

    # encoder_output
    encoder_outputs = Add()([enc_fw1, enc_bw1])
    state_h_1 = Add()([state_h_fw1, state_h_bw1])
    state_c_1 = Add()([state_c_fw1, state_c_bw1])

    encoder_states1 = [state_h_1, state_c_1]

    # encoder_model
    encoder_model = Model(inputs=encoder_input, outputs=[
        encoder_outputs, state_h_1, state_c_1])

    print("#4 decoder")
    # decoder for train

    # define layers
    decode_LSTM1 = LSTM(n_hidden,
                        name="decode_LSTM1",
                        return_sequences=True,
                        return_state=True)
    decoder_Dense = Dense(output_dim,
                          activation="softmax",
                          name="decoder_Dense")
    # deocoder
    decoder_inputs = Input(shape=(maxlen_d,), name="decoder_inputs")
    if use_dec_emb and not test:
        dec_emb = Embedding(output_dim=emb_dim,
                            input_dim=output_dim,
                            weights=[dec_emb_mat])(decoder_inputs)
    else:
        dec_emb = Embedding(output_dim=emb_dim,
                            input_dim=output_dim,
                            embeddings_initializer=uniform())(decoder_inputs)

    decoder_input = BatchNormalization(axis=-1)(dec_emb)
    dec_input = decoder_input
    decoder_lstm1, state_s_1, state_c_1 = decode_LSTM1(
        dec_input, initial_state=encoder_states1)

    decoder_outputs = decoder_Dense(decoder_lstm1)
    print("#5")

    model = Model(inputs=[encoder_input, decoder_inputs],
                  outputs=decoder_outputs)
    # model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam",metrics=["sparse_categorical_accuracy"])
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam", metrics=["categorical_accuracy"])

    print("#6")
    # decoder for generate translation
    decoder_state_input_h_1 = Input(
        shape=(n_hidden,), name='input_h_1')
    decoder_state_input_c_1 = Input(
        shape=(n_hidden,), name='input_c_1')

    # 　上のやつをまとめる
    decoder_states_inputs_1 = [
        decoder_state_input_h_1, decoder_state_input_c_1]

    decoder_states_inputs = [
        decoder_state_input_h_1, decoder_state_input_c_1]

    # LSTM
    decoder_lstm_1, state_h_1, state_c_1 = decode_LSTM1(
        dec_input, initial_state=decoder_states_inputs_1)

    decoder_states = [state_h_1, state_c_1]

    print("#7")
    # output
    decoder_outputs = decoder_Dense(decoder_lstm_1)

    # decoder model
    decoder_model = Model(
        [decoder_inputs]+decoder_states_inputs,
        [decoder_outputs]+decoder_states)

    return model, encoder_model, decoder_model
