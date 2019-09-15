import torch
import copy
import time
import heapq
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim
from utils.plot_hist import show_loss_plot
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
ref
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""


class Embedder(nn.Module):
    """
    return embedding
    """

    def __init__(self, config, emb=None, enc=True):
        super(Embedder, self).__init__()
        self.emb_dim = config["emb_dim"]
        self.hidden_size = config["n_hidden"]
        if enc:
            self.input_size = config["input_dim"]
        else:
            self.input_size = config["output_dim"]
        self.mask_token = config["mask_token"]

        if emb is None:
            self.embedding = nn.Embedding(
                self.input_size, self.emb_dim, padding_idx=self.mask_token)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb, freeze=False)

    def forward(self, x):
        return self.embedding(x)


class PositionalEndoder(nn.Module):
    """
    PositionalEndoder
    """

    def __init__(self, config):
        super(PositionalEndoder, self).__init__()
        self.hidden_size = config["emb_dim"]
        self.max_len = config["MAX_LENGTH"]
        self.pos_enc = self._build_pos_enc()

    def _build_pos_enc(self):
        pe = torch.zeros(self.max_len, self.hidden_size).to(device)
        for pos in range(self.max_len):
            for i in range(0, self.hidden_size, 2):
                pe[pos][i] = math.sin(pos/(10000**((2*i)/self.hidden_size)))
                pe[pos][i+1] = math.cos(pos /
                                        (10000**((2*(i+1))/self.hidden_size)))

        pe = pe.unsqueeze(0).to(device)  # (1,max_len,hidden)
        return pe

    def forward(self, x):
        """
        forward

        Parameters
        -------------------
                x:embedding (batch,max_len,emb_dim)

        Returns
        --------------------
        """

        x = x*math.sqrt(self.hidden_size)  # original imple of ref
        x = x+self.pos_enc[:, :x.size(1)]  # broadcasting
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.emb_dim = config["emb_dim"]
        self.heads = config["n_heads"]
        self.mask = mask
        self.max_len = config["MAX_LENGTH"]

        self.d_k = self.hidden_size//self.heads
        self.h = self.heads

        self.q_linear = nn.Linear(self.emb_dim, self.hidden_size)
        self.k_linear = nn.Linear(self.emb_dim, self.hidden_size)
        self.v_linear = nn.Linear(self.emb_dim, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.emb_dim)

    def _attention(self, q, k, v, hidden_size, mask=None):
        size = q.size(2)
        peak_mask = torch.from_numpy(
            np.triu(np.ones((1, size, size)), k=1).astype('uint8')).to(device)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hidden_size)
        if self.mask is not None:
            scores = scores.masked_fill(peak_mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).to(device)
        q = self.k_linear(q).view(bs, -1, self.h, self.d_k).to(device)
        v = self.k_linear(v).view(bs, -1, self.h, self.d_k).to(device)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self._attention(q, k, v, self.hidden_size, self.mask)
        concat = scores.transpose(1, 2).contiguous().view(
            bs, -1, self.hidden_size)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.emb_dim = config["emb_dim"]
        self.hidden_size = config["n_hidden"]
        self.linear_1 = nn.Linear(self.emb_dim, self.hidden_size)
        self.dropout = nn.Dropout()
        self.linear_2 = nn.Linear(self.hidden_size, self.emb_dim)

    def forward(self, x, is_train=True):
        x = self.linear_1(x)
        x = F.relu(x)
        if is_train:
            x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, config):
        super(Norm, self).__init__()
        self.hidden_size = config["emb_dim"]
        self.alpha = nn.Parameter(torch.ones(self.hidden_size)).to(device)
        self.bias = nn.Parameter(torch.zeros(self.hidden_size)).to(device)
        self.eps = 1e-6

    def forward(self, x):
        norm = self.alpha*(x-x.mean(dim=-1, keepdim=True)) / \
            (x.std(dim=-1, keepdim=True)+self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.norm_1 = Norm(config).to(device)
        self.norm_2 = Norm(config).to(device)
        self.attn = MultiHeadAttention(config).to(device)
        self.ff = FeedForward(config).to(device)
        self.dropout_1 = nn.Dropout()
        self.dropout_2 = nn.Dropout()

    def forward(self, x, is_train=True):
        x2 = self.norm_1(x)
        x_attn = self.attn(x2, x2, x2)
        if is_train:
            x_attn = self.dropout_1(x_attn)
        x = x + x_attn
        x2 = self.norm_2(x)
        x2 = self.ff(x2)
        if is_train:
            x2 = self.dropout_2(x2)
        x = x + x2
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.norm_1 = Norm(config).to(device)
        self.norm_2 = Norm(config).to(device)
        self.norm_3 = Norm(config).to(device)

        self.dropout_1 = nn.Dropout()
        self.dropout_2 = nn.Dropout()
        self.dropout_3 = nn.Dropout()

        self.attn_1 = MultiHeadAttention(config, mask=True).to(device)
        self.attn_2 = MultiHeadAttention(config).to(device)
        self.ff = FeedForward(config)

    def forward(self, x, e_outputs, is_train=True):
        x2 = self.norm_1(x)

        x_attn = self.attn_1(x2, x2, x2)
        if is_train:
            x_attn = self.dropout_1(x_attn)
        x = x + x_attn
        x2 = self.norm_2(x)
        x_attn2 = self.attn_2(x2, e_outputs, e_outputs)
        if is_train:
            x_attn2 = self.dropout_2(x_attn2)
        x = x + x_attn2
        x2 = self.norm_3(x)
        ff = self.ff(x2)
        if is_train:
            ff = self.dropout_3(ff)
        x = x + ff
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.N = config["n_layer"]
        self.embed = Embedder(config).to(device)
        self.pe = PositionalEndoder(config).to(device)

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(self.N)])
        self.norm = Norm(config).to(device)

    def forward(self, src, is_train=True):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, is_train)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.N = config["n_layer"]
        self.embed = Embedder(config, enc=False).to(device)
        self.pe = PositionalEndoder(config).to(device)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(self.N)])
        self.norm = Norm(config).to(device)

    def forward(self, trg, e_outputs, is_train=True):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, is_train)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.output_size = config["output_dim"]
        self.hidden_size = config["emb_dim"]
        self.encoder = Encoder(config).to(device)
        self.decoder = Decoder(config).to(device)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, src, trg, is_train=True):
        e_outputs = self.encoder(src, is_train)
        d_output = self.decoder(trg, e_outputs, is_train)
        output = self.out(d_output)
        output = self.softmax(output)
        return output


class Model():
    def __init__(self, config, enc_emb=None, dec_emb=None):
        self.learning_rate = config["learning_rate"]
        self.MAX_LENGTH = config["MAX_LENGTH"]
        self.EOS_token = config["EOS_token"]
        self.SOS_token = config["SOS_token"]
        self.n_hidden = config["n_hidden"]
        self.translate_length = config["translate_length"]
        self.val_size = config["val_size"]
        self.mask_token = config["mask_token"]
        self.param_dir = config["param_dir"]

        enc_emb = torch.FloatTensor(enc_emb).to(
            device) if not enc_emb is None else None
        dec_emb = torch.FloatTensor(dec_emb).to(
            device) if not dec_emb is None else None

        self.model = Transformer(config).to(device)
        self.criterion = nn.NLLLoss(ignore_index=self.mask_token).to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def _calc_loss(self, input_tensor, target_tensor):
        # input_tensor=(batch_size,length)
        self.optimizer.zero_grad()

        # input_tensor=(length,batch)
        # target_tensor=(length,batch)
        # taeget_tensorはshiftedの状態で入ってくる。
        dec_inp = target_tensor[:, :-1]
        targ = target_tensor[:, 1:].contiguous().view(-1).to(device)
        output = self.model(input_tensor, dec_inp)
        out = output.view(-1, output.size(-1)).to(device)

        # loss = self.criterion(output, target_tensor)
        loss = self.criterion(out, targ)
        return loss

    def train(self, input_tensor, target_tensor):
        loss = self._calc_loss(input_tensor, target_tensor)
        # back propagate
        if torch.isnan(loss):
            print("nan loss->", loss)

        loss.backward()
        self.optimizer.step()
        # loss_sum/batch_size
        return loss.item() / input_tensor.size()[0]

    def validation(self, input_tensor, target_tensor):
        with torch.no_grad():
            loss = self._calc_loss(input_tensor, target_tensor)
        return loss.item() / input_tensor.size()[0]

    def trainIters(self, src, trg, val_src, val_trg, save=False, load=False):
        data_train = [(torch.LongTensor(s), torch.LongTensor(t))
                      for s, t in zip(src, trg)]
        data_val = [(torch.LongTensor(s), torch.LongTensor(t))
                    for s, t in zip(val_src, val_trg)]
        train_loader = DataLoader(
            data_train, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(
            data_val, batch_size=self.batch_size, shuffle=True)
        print("train_size:{} - val_size:{}".format(len(data_train), len(data_val)))

        if load:
            self.load_model()
            print("pre_trained model is loaded")
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
            print("----------------")
            train_losses.append(epoch_loss/batch_cnt)
            val_losses.append(val_loss/val_bacth)
        show_loss_plot(train_losses, val_losses)
        if save:
            self.save_model()
            print("saved_model")

    def translate(self, src):
        input_tensor = torch.tensor(
            src, dtype=torch.long, device=device).view(1, -1).to(device)
        with torch.no_grad():
            e_outputs = self.model.encoder(input_tensor, is_train=False)
            outputs = torch.zeros(self.MAX_LENGTH).to(device)
            outputs[0] = torch.LongTensor([self.SOS_token])

            for i in range(1, self.MAX_LENGTH):
                dec_inp = torch.tensor(
                    outputs[:i], dtype=torch.long).view(1, -1).to(device)
                out = self.model.decoder(
                    dec_inp, e_outputs, is_train=False)  # (1,len,emb_dim)
                out = self.model.out(out)  # (1,len,out_voc)
                out = F.softmax(out, dim=2)
                val, ix = out[0][-1].data.topk(1)
                outputs[i] = int(ix[0])
                if ix[0] == self.EOS_token:
                    break
        return list(map(int, outputs[:i]))

    def translate_beam(self, src, beam_size=2):
        """
        beam seach
        return beam_size candidate
        """

        input_tensor = torch.tensor(
            src, dtype=torch.long, device=device).view(1, -1).to(device)
        with torch.no_grad():
            e_outputs = self.model.encoder(input_tensor, is_train=False)
            beam_outputs = [
                [-1, torch.zeros(self.MAX_LENGTH).to(device)]]
            beam_outputs[0][1][0] = torch.LongTensor([self.SOS_token])
            for i in range(1, self.MAX_LENGTH):
                beam_cand = []
                for score, cand in beam_outputs:
                    dec_inp = torch.tensor(
                        cand[:i], dtype=torch.long).view(1, -1).to(device)
                    out = self.model.decoder(
                        dec_inp, e_outputs, is_train=False)
                    out = self.model.out(out)
                    out = F.softmax(out, dim=2)  # (1,1,out_voc)であってる?
                    res = out[0][-1].data.topk(beam_size)
                    for b in range(beam_size):
                        next_cand = copy.deepcopy(cand)
                        next_cand[i] = int(res[1][b])
                        heapq.heappush(
                            beam_cand, [score*float(res[0][b]), next_cand])

                beam_outputs = []
                for _ in range(beam_size):
                    beam_outputs.append(heapq.heappop(beam_cand))

            ret = []
            for _, b in beam_outputs:
                ret.append(list(map(int, b)))
            return ret

    def save_model(self):
        torch.save(self.model.state_dict(), self.param_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.param_dir))
