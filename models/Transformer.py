import torch
import time
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils.plot_hist import show_loss_plot, show_attention
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
        print("x->", x.size())
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
        x = x+self.pos_enc.repeat(x.size(0), 1, 1)  # FIXME もっと良い実装があるはず
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, config, dropout=0.1):  # FIXME paramをconfigにまとめる
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.emb_dim = config["emb_dim"]
        self.d_k = self.hidden_size//heads
        self.h = heads

        self.q_linear = nn.Linear(self.emb_dim, self.hidden_size)
        self.k_linear = nn.Linear(self.emb_dim, self.hidden_size)
        self.v_linear = nn.Linear(self.emb_dim, self.hidden_size)

        self.drop_out = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.hidden_size, self.emb_dim)

    def _attention(self, q, k, v, hidden_size):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hidden_size)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.hidden_size).to(device)
        q = self.k_linear(q).view(bs, -1, self.h, self.hidden_size).to(device)
        v = self.k_linear(v).view(bs, -1, self.h, self.hidden_size).to(device)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self._attention(q, k, v, self.hidden_size)
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

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
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
    def __init__(self, config, heads):  # FIXME congifにまとめる
        super(EncoderLayer, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.norm_1 = Norm(config).to(device)
        self.norm_2 = Norm(config).to(device)
        self.attn = MultiHeadAttention(heads, config).to(device)
        self.ff = FeedForward(config).to(device)
        self.dropout_1 = nn.Dropout()
        self.dropout_2 = nn.Dropout()

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config, heads):
        super(DecoderLayer, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.norm_1 = Norm(config).to(device)
        self.norm_2 = Norm(config).to(device)
        self.norm_3 = Norm(config).to(device)

        self.dropout_1 = nn.Dropout()
        self.dropout_2 = nn.Dropout()
        self.dropout_3 = nn.Dropout()

        self.attn_1 = MultiHeadAttention(heads, config).to(device)
        self.attn_2 = MultiHeadAttention(heads, config).to(device)
        self.ff = FeedForward(config)

    def forward(self, x, e_outputs):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, config, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.embed = Embedder(config).to(device)
        self.pe = PositionalEndoder(config).to(device)

        self.layers = nn.ModuleList(
            [EncoderLayer(config, heads) for _ in range(N)])
        self.norm = Norm(config).to(device)

    def forward(self, src):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.embed = Embedder(config, enc=False).to(device)
        self.pe = PositionalEndoder(config).to(device)
        # self.layers = get_clones(DecoderLayer(config, heads), N)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, heads) for _ in range(N)])
        self.norm = Norm(config).to(device)

    def forward(self, trg, e_outputs):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config, N, heads):
        super(Transformer, self).__init__()
        self.output_size = config["output_dim"]
        self.hidden_size = config["emb_dim"]
        self.encoder = Encoder(config, N, heads).to(device)
        self.decoder = Decoder(config, N, heads).to(device)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, src, trg):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs)
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

        enc_emb = torch.FloatTensor(enc_emb).to(
            device) if not enc_emb is None else None
        dec_emb = torch.FloatTensor(dec_emb).to(
            device) if not dec_emb is None else None

        self.model = Transformer(config, 1, 10).to(device)
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
        # input_tensor = input_tensor.transpose(0, 1)
        # target_tensor = target_tensor.transpose(0, 1)

        output = self.model(input_tensor, target_tensor)
        # print("out->", output.size())  # (batch,len,out)
        # print("target->", target_tensor.size())  # (batch,len)
        out = output.view(-1, output.size(-1)).to(device)
        targ = target_tensor.view(-1).to(device)

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

    def trainIters(self, src, trg, val_src, val_trg):
        data_train = [(torch.LongTensor(s), torch.LongTensor(t))
                      for s, t in zip(src, trg)]
        data_val = [(torch.LongTensor(s), torch.LongTensor(t))
                    for s, t in zip(val_src, val_trg)]
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
            print("----------------")
            train_losses.append(epoch_loss/batch_cnt)
            val_losses.append(val_loss/val_bacth)
        show_loss_plot(train_losses, val_losses)

    def translate(self, src):
        input_tensor = torch.tensor(
            src, dtype=torch.long).view(1, -1).to(device)
        with torch.no_grad():
            e_outputs = self.model.encoder(input_tensor)
            outputs = torch.zeros(self.MAX_LENGTH).to(device)  # FIXME errorでる
            #outputs[0] = torch.LongTensor([self.SOS_token])
            print("outputs->", outputs[:2].view(1, -1).size())

            for i in range(1, self.MAX_LENGTH+1):
                out = self.model.decoder(
                    outputs[:i].view(1, -1).to(device), e_outputs)
                print("out->", out.size())
