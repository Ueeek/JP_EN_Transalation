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

    def __init__(self, config):
        super(EncoderRnn, self).__init__()
        self.emb_dim = config["emb_dim"]
        self.hidden_size = config["n_hidden"]
        self.input_size = config["input_dim"]
        self.mask_token = config["mask_token"]
        self.emb_dim = config["emb_dim"]

        self.embedding = nn.Embedding(
            self.input_size, self.emb_dim, padding_idx=self.mask_token)
        self.gru = nn.GRU(
            self.emb_dim, self.hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)

    def forward(self, input, batch_size, hidden):
        """"
        Param:
        -------------
        input: tensor(batch_size)
        batch_size:int
        hidden: tensor(4,batch,hidden)

        Returns:
        ---------------
        output: tensor(1,batch,hidden)
        hidden: tensor(4,batch,hidden)

        """
        # embedded=(1,batch,emb_dim)
        embedded = self.embedding(input).view(1, batch_size, self.emb_dim)
        # gru_output=(1,batch,hidden_dim*2)
        # hidden_output=(4,batch,hidden_dim)
        gru_output, hidden_output = self.gru(embedded, hidden)

        # add forward and backword
        encoder_output = gru_output[:, :, :self.hidden_size] + \
            gru_output[:, :, self.hidden_size:]
        return encoder_output, hidden_output

    def initHidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size).to(device)


class DecoderRnn(nn.Module):
    """
    decoder RNN with attention
    """

    def __init__(self, config):
        super(DecoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.output_size = config["output_dim"]
        self.max_length = config["MAX_LENGTH"]
        self.mask_token = config["mask_token"]
        self.emb_dim = config["emb_dim"]

        self.embedding = nn.Embedding(
            self.output_size, self.emb_dim, padding_idx=self.mask_token)
        self.gru = nn.GRU(self.emb_dim, self.hidden_size,
                          num_layers=2, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, batch_size, hidden):
        """"
        Param:
        -------------
        input: tensor(batch_size)
        batch_size:int
        hidden: tensor(4,batch,hidden)

        Returns:
        ---------------
        output: tensor(1,batch,output_size)
        hidden: tensor(4,batch,hidden)

        """
        # embedded = (1,batch,emb_dim)
        embedded = self.embedding(input).view(
            1, batch_size, self.emb_dim).to(device)
        gru_in = F.relu(embedded)
        # gru_out = (1,batch.hidden*2)
        # hidden_out = (4,batch,hidden)
        gru_out, hidden_out = self.gru(gru_in, hidden)

        # gru_output=(1,batch,hidden)
        gru_output = gru_out[:, :, :self.hidden_size] + \
            gru_out[:, :, self.hidden_size:]
        output = self.out(gru_output)
        decoder_output = self.softmax(output)
        return decoder_output, hidden_out

    def _initHidden(self, batch_size):
        return torch(4, batch_size, self.hidden_size).to(device)


class Seq2Seq:
    def __init__(self, config):
        self.learning_rate = config["learning_rate"]
        self.MAX_LENGTH = config["MAX_LENGTH"]
        self.EOS_token = config["EOS_token"]
        self.SOS_token = config["SOS_token"]
        self.n_hidden = config["n_hidden"]
        self.translate_length = config["translate_length"]
        self.val_size = config["val_size"]
        self.mask_token = config["mask_token"]

        self.encoder = EncoderRnn(config).to(device)
        self.decoder = DecoderRnn(config).to(device)
        self.criterion = nn.NLLLoss(ignore_index=self.mask_token).to(device)
        self. encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=self.learning_rate)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def calc_loss(self, input_tensor, target_tensor):
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
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], batch_size, encoder_hidden)

        # decoder
        decoder_input = torch.LongTensor(
            [self.SOS_token]*batch_size).to(device)
        decoder_hidden = encoder_hidden
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, batch_size, decoder_hidden)

            tmp_loss = self.criterion(decoder_output[0], target_tensor[di])
            # print("tmp_loss->", tmp_loss)
            decoder_input = target_tensor[di]  # teacher forcing
            loss += tmp_loss
        return loss

    def train(self, input_tensor, target_tensor):
        loss = self.calc_loss(input_tensor, target_tensor)
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
            loss = self.calc_loss(input_tensor, target_tensor)
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

    # FixME tensorの形がやばそう。batchnormのとこで、2次元になっててerror(batch　処理にした方が楽?)
    def translate(self, input_tensor):
        """
        parameter
        ------------
            input_temnsor: (batch,length)
        """
        with torch.no_grad():
            batch_size = input_tensor.size()[0]
            input_length = input_tensor.size()[1]

            input_tensor = input_tensor.transpose(0, 1)
            # Encoder
            encoder_hidden = self.encoder.initHidden(batch_size)
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], batch_size, encoder_hidden)

            # Decoder
            decoder_input = torch.tensor(
                [self.SOS_token]*batch_size).to(device)

            decoder_hidden = encoder_hidden

            decoded_ids = [[] for _ in range(batch_size)]
            for di in range(self.translate_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, batch_size, decoder_hidden)
                # dec_out = (1,batch,out_size)
                dec_out = decoder_output[0].argmax(dim=1)
                for i in range(batch_size):
                    decoded_ids[i].append(dec_out[i].item())
        return decoded_ids

    def translateIter(self, src):
        input_tensor = torch.tensor(
            src, dtype=torch.long).view(len(src), -1).to(device)
        return self.translate(input_tensor)
