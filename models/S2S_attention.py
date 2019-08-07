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
    encoder + batch_norm+dropout
    """

    def __init__(self, config, emb=None):
        super(EncoderRnn, self).__init__()
        self.emb_dim = config["emb_dim"]
        self.hidden_size = config["n_hidden"]
        self.input_size = config["input_dim"]
        self.mask_token = config["mask_token"]
        self.emb_dim = config["emb_dim"]
        self.batch_norm = nn.BatchNorm1d(self.emb_dim)

        if emb is None:
            self.embedding = nn.Embedding(
                self.input_size, self.emb_dim, padding_idx=self.mask_token)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb, freeze=False)
        self.gru = nn.GRU(
            self.emb_dim, self.hidden_size, num_layers=2, bidirectional=True, dropout=0.3)

    def forward(self, input, batch_size, hidden):
        """"
        Param:
        -------------
        input: tensor(batch_size)
        batch_size:int
        hidden: tensor(4,batch,hidden)

        Returns:
        ---------------
        output: tensor(1,batch,hidden*2)
        hidden: tensor(4,batch,hidden)

        """
        # embedded=(batch,emb_dim)
        # batch_norm=(1,batch,emb_dim)
        embedded = self.embedding(input)
        batch_norm = self.batch_norm(embedded).view(
            1, batch_size, self.emb_dim).to(device)
        # gru_output=(1,batch,hidden_dim*2)
        # hidden_output=(4,batch,hidden_dim)
        gru_output, hidden_output = self.gru(batch_norm, hidden)

        # add forward and backword
        # encoder_output = gru_output[:, :, :self.hidden_size] + \
        #   gru_output[:, :, self.hidden_size:]
        encoder_output = gru_output
        return encoder_output, hidden_output

    def initHidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size).to(device)


class Attention(nn.Module):
    """
    Attention Layer
    """

    def __init__(self, config):
        super(Attention, self).__init__()
        self.hidden_size = config["n_hidden"]

    def forward(self, encoder_outputs, decoder_gru_output):
        """
        Parameters
        -------------------
                encoder_outputs:(input_len,batcj,hidden*2)
                decoder_gru_output:(1,batch,hidden*2)

        Returns     
                attmsum: (1,batch,hidden*2)
        --------------------
        """

        input_len, batch_size, _ = encoder_outputs.size()

        # repeat(1,batch,hidden*2)->(input_len,batch_size,hidden*2)
        rep_gru_out = decoder_gru_output.repeat(input_len, 1, 1)
        # product(input_len,batch_size,hidden*2)
        weight_product = rep_gru_out * encoder_outputs

        # weight_sum(input_len,batch)
        weight_sum = weight_product.sum(dim=2)

        # weight_softmax(input_len,batch)
        weight_softmax = F.softmax(weight_sum, dim=0).view(
            input_len, batch_size, 1)

        #output = (1,batch,output_size)
        # repeat(input_len,batch_size,hidden*2)
        rep_soft = weight_softmax.repeat(1, 1, self.hidden_size*2)
        # product
        # attn_mul(input_len,batch,hidden*2)
        attn_mul = rep_soft * encoder_outputs
        #attn_sum (batch,hidden*2)
        attn_sum = attn_mul.sum(dim=0)
        return attn_sum.view(1, batch_size, self.hidden_size*2).to(device)


class AttnDecoderRnn(nn.Module):
    """
    decoder RNN + attention
    """

    def __init__(self, config, emb=None):
        super(AttnDecoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.output_size = config["output_dim"]
        self.max_length = config["MAX_LENGTH"]
        self.mask_token = config["mask_token"]
        self.emb_dim = config["emb_dim"]

        self.batch_norm = nn.BatchNorm1d(self.emb_dim)
        if emb is None:
            self.embedding = nn.Embedding(
                self.output_size, self.emb_dim, padding_idx=self.mask_token)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb, freeze=False)
        self.gru = nn.GRU(self.emb_dim, self.hidden_size,
                          num_layers=2, bidirectional=True, dropout=0.3)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.attention = Attention(config)

    def forward(self, input, batch_size, hidden, encoder_outputs):
        """"
        Param:
        -------------
        input: tensor(batch_size)
        batch_size:int
        hidden: tensor(4,batch,hidden)
        encoder_outputs:()

        Returns:
        ---------------
        output: tensor(1,batch,output_size)
        hidden: tensor(4,batch,hidden)

        """
        # embedded = (batch,emb_dim)
        # batch_norm = (1,batch,emb_dim)
        embedded = self.embedding(input)
        batch_norm = self.batch_norm(embedded).view(
            1, batch_size, self.emb_dim).to(device)
        gru_in = F.relu(batch_norm)
        # gru_out = (1,batch.hidden*2)
        # hidden_out = (4,batch,hidden)
        gru_out, hidden_out = self.gru(gru_in, hidden)

        # attn =(1,batch,hidden*2)
        attn = self.attention(encoder_outputs, gru_out)
        # lin =(1,batch,hidden)
        lin = self.linear(attn)
        lin = F.relu(lin)
        #output = (1,batch,output_size)
        output = self.out(lin)
        decoder_output = self.softmax(output)
        return decoder_output, hidden_out  # ,attn

    def _initHidden(self, batch_size):
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
        self.decoder = AttnDecoderRnn(config, dec_emb).to(device)
        self.criterion = nn.NLLLoss(ignore_index=self.mask_token).to(device)
        self. encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=self.learning_rate)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def _calc_loss(self, input_tensor, target_tensor):
        # input_tensor=(batch_size,length)
        batch_size = input_tensor.size()[0]
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # input_tensor=(length,batch)
        # target_tensor=(length,batch)
        input_tensor = input_tensor.transpose(0, 1)
        target_tensor = target_tensor.transpose(0, 1)

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        # encoder
        loss = 0
        # encoder_hidden =(4,batch,hidden)
        encoder_hidden = self.encoder.initHidden(batch_size)
        encoder_outputs = torch.zeros(
            input_length, batch_size, self.n_hidden*2).to(device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        # decoder
        decoder_input = torch.LongTensor(
            [self.SOS_token]*batch_size).to(device)
        decoder_hidden = encoder_hidden
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, batch_size, decoder_hidden, encoder_outputs)

            tmp_loss = self.criterion(decoder_output[0], target_tensor[di])
            # print("tmp_loss->", tmp_loss)
            decoder_input = target_tensor[di]  # teacher forcing
            loss += tmp_loss
        return loss

    def train(self, input_tensor, target_tensor):
        loss = self._calc_loss(input_tensor, target_tensor)
        # back propagate
        if torch.isnan(loss):
            print("nan loss->", loss)

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
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

    def _translate(self, input_tensor):
        """
        parameter
        ------------
            input_temnsor: (batch,length)
        """
        with torch.no_grad():
            batch_size = input_tensor.size()[0]
            input_length = input_tensor.size()[1]

            input_tensor = input_tensor.transpose(0, 1)
            encoder_outputs = torch.zeros(
                input_length, batch_size, self.n_hidden*2).to(device)
            # Encoder
            encoder_hidden = self.encoder.initHidden(batch_size)
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], batch_size, encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]

            # Decoder
            decoder_input = torch.tensor(
                [self.SOS_token]*batch_size).to(device)

            decoder_hidden = encoder_hidden

            decoded_ids = [[] for _ in range(batch_size)]
            for di in range(self.translate_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, batch_size, decoder_hidden, encoder_outputs)
                # dec_out = (1,batch,out_size)
                dec_out = decoder_output[0].argmax(dim=1)
                for i in range(batch_size):
                    decoded_ids[i].append(dec_out[i].item())
                decoder_input = dec_out
        return decoded_ids

    def translateIter(self, src):
        input_tensor = torch.tensor(
            src, dtype=torch.long).view(len(src), -1).to(device)
        return self._translate(input_tensor)
