import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils.plot_hist import showPlot
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

        if emb is None:
            self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        else:
            print("emb_enc", emb.size())
            print("int->", self.input_size)
            print("hid->", self.hidden_size)
            self.embedding = nn.Embedding.from_pretrained(emb)
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(
            1, batch_size, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size).to(device)


class DecoderRnn(nn.Module):
    """
    decoder
    """

    def __init__(self, config, emb):
        super(DecoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.output_size = config["output_dim"]

        if emb is None:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        else:
            print("emb_dec", emb.size())
            print("int->", self.output_size)
            print("hid->", self.hidden_size)
            self.embedding = nn.Embedding.from_pretrained(emb)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,
                          num_layers=2, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        output = self.embedding(input).view(1, batch_size, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size).to(device)


class Seq2Seq:
    def __init__(self, config, enc_emb=None, dec_emb=None):
        self.print_every = 10
        self.plot_epoch = 10
        self.learning_rate = config["learning_rate"]
        self.MAX_LENGTH = config["MAX_LENGTH"]
        self.EOS_token = config["EOS_token"]
        self.SOS_token = config["SOS_token"]
        self.n_hidden = config["n_hidden"]
        self.translate_length = config["translate_length"]

        enc_emb = torch.FloatTensor(enc_emb).to(device)
        dec_emb = torch.FloatTensor(dec_emb).to(device)
        print("enc_emb_seq", enc_emb.size())
        print("dec_emb_size", dec_emb.size())
        self.encoder = EncoderRnn(config, enc_emb).to(device)
        self.decoder = DecoderRnn(config, dec_emb).to(device)
        self.criterion = nn.NLLLoss().to(device)
        self. encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_opimizer = optim.SGD(
            self.decoder.parameters(), lr=self.learning_rate)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def train(self, input_tensor, target_tensor):

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
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, batch_size, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)
            loss += self.criterion(decoder_output, target_tensor[di])
            # print("loss->", type(loss))
            decoder_input = target_tensor[di]

        # back propagate
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_opimizer.step()
        return loss.item() / target_length

    def trainIters(self, src, trg):

        data = [(torch.LongTensor(s), torch.LongTensor(t))
                for s, t in zip(src, trg)]
        train_loader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True)

        # batchを作る
        plot_losses = []
        plot_loss_total = 0

        for epoch in range(self.epochs):
            print("epoch{} start".format(epoch+1))
            start = time.time()
            print_loss_total = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = self.train(batch_x, batch_y)
                print_loss_total += loss
                plot_loss_total += loss

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total/self.print_every
                print_loss_total = 0
                print("loss_av in eopch{}=> {}".format(epoch, print_loss_avg))
                print("time->", time.time() - start)

            if epoch % self.plot_epoch == 0:
                plot_loss_avg = plot_loss_total/self.plot_epoch
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        showPlot(plot_losses)

    def translate(self, input):
        ids = torch.tensor(input, dtype=torch.long).view(-1, 1).to(device)
        with torch.no_grad():
            input_length = ids.size()[0]
            encoder_hidden = self.encoder.initHidden(1)
            encoder_outputs = torch.zeros(
                self.MAX_LENGTH, 1, self.n_hidden).to(device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    ids[ei], 1, encoder_hidden)
                encoder_outputs[ei] += encoder_output[0]

            decoder_input = torch.tensor([[self.SOS_token]]).to(device)

            decoder_hidden = encoder_hidden

            decoded_ids = []
            for di in range(self.translate_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, 1, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                if topi.item() == self.EOS_token:
                    decoded_ids.append(self.EOS_token)
                    break
                else:
                    decoded_ids.append(topi.item())
                decoder_input = topi.squeeze().detach()
            return decoded_ids
