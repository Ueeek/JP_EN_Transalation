import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils.plot_hist import show_loss_plot
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRnn(nn.Module):
    """
    encoder
    """

    def __init__(self, config, emb):
        super(EncoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.input_size = config["input_dim"]
        self.mask_token = config["mask_token"]

        if emb is None:
            self.embedding = nn.Embedding(
                self.input_size, self.hidden_size, padding_idx=self.mask_token)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                emb, freeze=False, padding_idx=self.mask_token)

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(
            1, batch_size, self.hidden_size)
        # output = self.batch_norm(embedded[0]).view(
        #    1, batch_size, -1).to(device)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size).to(device)


class Attention(nn.Module):
    """
    calc Attention
    """

    def __init__(self, config):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_gru_out):
        # encoder_outputs=(input_len,batch,hidden)
        # decoder_gru_out=(1,batch,hidden)
        inp_len, batch_len, hidden_len = encoder_outputs.size()

        # rep_gru_out = (input_len,batch,hidden)
        rep_gru_out = decoder_gru_out.repeat(inp_len, 1, 1)
        # weight_prod = (input_len,batch,hidden)
        weight_prod = rep_gru_out*encoder_outputs
        # weight_sum = (inp_len,batch)
        weight_sum = weight_prod.sum(dim=2)
        # weight_softmax = (inp_len,batch,1)
        weight_softmax = F.softmax(weight_sum, dim=0).view(
            inp_len, batch_len, 1).to(device)

        # soft_rep = (inp_len,bacth,hidden)
        soft_rep = weight_softmax.repeat(1, 1, hidden_len)
        # attn_mul = (inp_len,batch_len,hidden)
        attn_mul = soft_rep*encoder_outputs
        # attnsum = (batch,hidden)
        attn_sum = attn_mul.sum(dim=0)

        return attn_sum


class AttentionDecoderRnn(nn.Module):
    """
    decoder RNN with attention
    """

    def __init__(self, config, emb):
        super(AttentionDecoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.output_size = config["output_dim"]
        self.max_length = config["MAX_LENGTH"]
        self.attention_layer = Attention(config)
        self.mask_token = config["mask_token"]

        if emb is None:
            self.embedding = nn.Embedding(
                self.output_size, self.hidden_size, padding_idx=self.mask_token)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                emb, freeze=False, padding_idx=self.mask_token)

        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,
                          num_layers=2, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden, encoder_outputs):
        # input->(batch_size) 前のdecoderの結果的な(teacher forcingのtargrt)
        # hidden -> (4,batch_size,hidden_size)
        # encoder_outputs -> (inp_len,batch,hidden)

        # embedded = (1,batch,hidden)
        embedded = self.embedding(input).view(
            1, batch_size, self.hidden_size).to(device)
        # gru_in = (1,batch,hidden)
        # embedded = self.batch_norm(embedded[0]).view(
        #   1, batch_size, -1).to(device)
        gru_in = F.relu(embedded)
        # gru_out = (1,batch.hidden*2)
        # hidden_out = (4,batch,hidden)
        gru_out, hidden_out = self.gru(gru_in, hidden)
        # gru_out_linear=(1,batch,hidden) <- (1,batch,hidden*2)
        gru_out_linear = self.linear(gru_out)
        # attn=(batch,hidden)
        attn = self.attention_layer(encoder_outputs, gru_out_linear)
        # attn = gru_out_linear[0]
        # output->(batch,output_size)
        output = self.out(attn)
        output = self.softmax(output)
        return output, hidden_out, attn

    def initHidden(self, batch_size):
        return torch(4, batch_size, self.hidden_size).to(device)


class Seq2Seq:
    def __init__(self, config, enc_emb=None, dec_emb=None):
        self.learning_rate = config["learning_rate"]
        self.MAX_LENGTH = config["MAX_LENGTH"]
        self.EOS_token = config["EOS_token"]
        self.SOS_token = config["SOS_token"]
        self.n_hidden = config["n_hidden"]
        self.translate_length = config["translate_length"]
        self.val_size = config["val_size"]
        self.mask_token = config["mask_token"]

        enc_emb = torch.FloatTensor(enc_emb).to(
            device) if not enc_emb is None else None
        dec_emb = torch.FloatTensor(dec_emb).to(
            device) if not dec_emb is None else None
        self.encoder = EncoderRnn(config, enc_emb).to(device)
        self.decoder = AttentionDecoderRnn(config, dec_emb).to(device)
        self.criterion = nn.NLLLoss(ignore_index=self.mask_token).to(device)
        self. encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_opimizer = optim.SGD(
            self.decoder.parameters(), lr=self.learning_rate)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def calc_loss(self, input_tensor, target_tensor):
        # input_tensor=(batch_size,leng)
        batch_size = input_tensor.size()[0]
        # print("batch_size->", input_tensor.size())

        encoder_hidden = self.encoder.initHidden(batch_size)

        self.encoder_optimizer.zero_grad()
        self.decoder_opimizer.zero_grad()
        input_tensor = input_tensor.transpose(0, 1)
        target_tensor = target_tensor.transpose(0, 1)

        input_length = input_tensor.size()[0]
        # print(input_tensor.size())
        target_length = target_tensor.size()[0]

        # encoder
        encoder_outputs = torch.zeros(
            self.MAX_LENGTH, batch_size, self.n_hidden).to(device)
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        # decoder
        decoder_input = torch.LongTensor(
            [self.SOS_token]*batch_size).to(device)
        decoder_hidden = encoder_hidden.to(device)
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, batch_size, decoder_hidden, encoder_outputs)

            tmp_loss = self.criterion(decoder_output, target_tensor[di])
            # print("tmp_loss->", tmp_loss)
            decoder_input = target_tensor[di]
            loss += tmp_loss
        return loss

    def train(self, input_tensor, target_tensor):
        loss = self.calc_loss(input_tensor, target_tensor)
        # back propagate
        if torch.isnan(loss):
            print("nan loss->", loss)

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_opimizer.step()
        return loss.item() / input_tensor.size()[0]

    def validation(self, input_tensor, target_tensor):
        with torch.no_grad():
            loss = self.calc_loss(input_tensor, target_tensor)
        return loss.item() / input_tensor.size()[0]

    def trainIters(self, src, trg):
        val_size = self.val_size
        data_train = [(torch.LongTensor(s), torch.LongTensor(t))
                      for s, t in zip(src[:-val_size], trg[:-val_size])]
        data_val = [(torch.LongTensor(s), torch.LongTensor(t))
                    for s, t in zip(src[-val_size:], trg[-val_size:])]
        train_loader = DataLoader(
            data_train, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(
            data_val, batch_size=self.batch_size, shuffle=True)
        print("train_size:{} - val_size:{}".format(len(data_train), len(data_val)))

        # batchを作る
        train_losses = []
        val_losses = []
        for epoch in range(self.epochs):
            start = time.time()
            epoch_loss = 0
            batch_cnt = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = self.train(batch_x, batch_y)
                epoch_loss += loss
                batch_cnt += 1

            # calc validation loss
            val_loss = 0
            val_bacth = 0
            for batch_x, batch_y in val_loader:
                val_bacth += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_loss += self.validation(batch_x, batch_y)
            print("Epoch {}/{}".format(epoch+1, self.epochs))
            print("{:.0f} - loss: {:.5f} - val-loss: {:.5f}".format(time.time() -
                                                                    start, epoch_loss/batch_cnt, val_loss/val_bacth))
            # print("loss->{}, val_loss->{}".format(epoch_loss, val_loss))
            # print("epoch{}. time->{}".format(epoch+1, time.time()-start))
            # print("loss->{}, val_loss->{}".format(epoch_loss/batch_cnt, val_loss/val_bacth))
            print("----------------")
            train_losses.append(epoch_loss/batch_cnt)
            val_losses.append(val_loss/val_bacth)
        show_loss_plot(train_losses, val_losses)

    # FixME tensorの形がやばそう。batchnormのとこで、2次元になっててerror(batch　処理にした方が楽?)
    def translate(self, input):
        # ids=(input_len,1)
        ids = torch.tensor(input, dtype=torch.long).view(-1, 1).to(device)
        with torch.no_grad():
            input_length = ids.size()[0]
            encoder_hidden = self.encoder.initHidden(1)
            encoder_outputs = torch.zeros(
                self.MAX_LENGTH, 1, self.n_hidden).to(device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    ids[ei], 1, encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]

            decoder_input = torch.tensor([self.SOS_token]*1).to(device)

            decoder_hidden = encoder_hidden

            decoded_ids = []
            # decoder_attentions = torch.zeros(self.MAX_LENGTH, self.MAX_LENGTH)
            for di in range(self.translate_length):
                #print("dec_inp->", decoder_input.size())
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, 1, decoder_hidden, encoder_outputs)
                # decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.topk(1)
                #print("topv-", topv, topi.item())
                #print("topv,toppi->", decoder_output.topk(2))
                #print("topv,toppi without data->", decoder_output.data.topk(2))
                if topi.item() == self.EOS_token:
                    decoded_ids.append(self.EOS_token)
                    break
                else:
                    decoded_ids.append(topi.item())
                #decoder_input = topi.squeeze().detach()
                decoder_input = topi.view(1).to(device)
        return decoded_ids
