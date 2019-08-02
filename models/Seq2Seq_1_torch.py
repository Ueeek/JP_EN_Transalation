import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils.plot_hist import showPlot

device = torch.device("cude" if torch.cuda.is_available() else "cpu")


class EncoderRnn(nn.Module):
    """
    encoder
    """

    def __init__(self, config):
        super(EncoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.input_size = config["max_features"]

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lstm = nn.GRU(
            self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRnn(nn.Module):
    """
    decoder
    """

    def __init__(self, config):
        super(DecoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.output_size = config["max_features"]

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Seq2Seq:
    def __init__(self,config):
        self.print_every=10
        self.plot_every=100
        self.learning_rate = config["learning_rate"]
        self.MAX_LENGTH = config["MAX_LENGTH"]
        self.EOS_token = config["EOS_token"]
        self.SOS_token = config["SOS_token"]
        self.n_hidden = config["n_hidden"]
        self.translate_length = config["translate_length"]
        self.encoder = EncoderRnn(config)
        self.decoder = DecoderRnn(config)
        self.criterion = nn.NLLLoss()
        self. encoder_optimizer = optim.SGD(self.encoder.parameters(),lr = self.learning_rate)
        self.decoder_opimizer = optim.SGD(self.decoder.parameters(),lr=self.learning_rate)
        
    def train(self,input_tensor,target_tensor):
        
        encoder_hidden = self.encoder.initHidden()
        
        self.encoder_optimizer.zero_grad()
        self.decoder_opimizer.zero_grad()
        
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        #encoder
        encoder_outputs = torch.zeros(self.MAX_LENGTH,self.n_hidden,device=device)
        loss = 0
        for ei in range(input_length):
            encoder_output,encoder_hidden = self.encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] = encoder_output[0,0]
            
        # decoder
        decoder_input = torch.tensor([[self.SOS_token]],device=device)
        decoder_hidden = encoder_hidden
        for di in range(target_length):
            decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            loss += self.criterion(decoder_output,target_tensor[di])
            decoder_input = target_tensor[di]
        
        # back propagate
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_opimizer.step()
        return loss.item() / target_length
    
    def trainIters(self,input,target,n_iters):
        src= [torch.tensor(v,dtype=torch.long,device=device).view(-1,1) for v in input]
        trg= [torch.tensor(v,dtype=torch.long,device=device).view(-1,1) for v in target]
        plot_losses=[]
        print_loss_total=0
        plot_loss_total=0
        
        encoder_optimizer = optim.SGD(self.encoder.parameters(),lr = self.learning_rate)
        decoder_opimizer = optim.SGD(self.decoder.parameters(),lr=self.learning_rate)
        
        for iter in range(1,n_iters+1):
            input_tensor=src[iter-1]
            target_tensor=trg[iter-1]
            loss =self.train(input_tensor,target_tensor)
            print_loss_total += loss
            plot_loss_total += loss
            
            if iter %self.print_every==0:
                print_loss_avg = print_loss_total/self.print_every
                print_loss_total=0
                print("loss_av->",print_loss_avg)
            
            if iter % self.plot_every==0:
                plot_loss_avg = plot_loss_total/self.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        showPlot(plot_losses)
        
    def translate(self,input):
        ids = torch.tensor(input,dtype=torch.long,device=device).view(-1,1)
        with torch.no_grad():
            input_length = ids.size()[0]
            encoder_hidden = self.encoder.initHidden()
            encoder_outputs = torch.zeros(self.MAX_LENGTH,self.n_hidden,device=device)
            
            
            for ei in range(input_length):
                encoder_output,encoder_hidden = self.encoder(ids[ei],encoder_hidden)
                encoder_outputs[ei] += encoder_output[0,0]
                
            decoder_input = torch.tensor([[self.SOS_token]],device=device)
            
            decoder_hidden = encoder_hidden
            
            decoded_ids=[]
            for di in range(self.translate_length):
                decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
                topv,topi = decoder_output.topk(1)
                if topi.item() == self.EOS_token:
                    decoded_ids.append(self.EOS_token)
                    break
                else:
                    decoded_ids.append(topi.item())
                decoder_input = topi.squeeze().detach()
            return decoded_ids
            