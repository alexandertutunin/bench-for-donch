from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Linear(hidden_size, hidden_size // 2) #[batchsize, timestamps, features]
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.relu3 = nn.ReLU()

        self.out = nn.Linear(hidden_size // 8, output_size)

    def forward(self, x):  # x: [B, T, H]

        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.out(x)

        return x

class PureRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, rnn_layers = 1):
        super(PureRNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=rnn_layers)
        self.fc = MLP(hidden_size=hidden_size,
                        output_size=output_size)
        
    def forward(self, x):
        
        rnn_out, _ = self.rnn(x)

        out = self.fc(rnn_out)  # (batch_size, sequence_length, output_size=1)
        out = out.squeeze(-1) # (batch_size, sequence_length)
        return out
    

class MaxMapRNN(nn.Module):
    def __init__(self, kernel_size, input_size, hidden_size, output_size=1, rnn_layers = 1):
        super(MaxMapRNN, self).__init__()
        
        reduced_size = input_size // kernel_size

        self.maxmap = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)
        self.rnn = nn.RNN(reduced_size, hidden_size, batch_first=True, num_layers=rnn_layers)
        
        self.fc = MLP(hidden_size=hidden_size,
                        output_size=output_size)
        
    def forward(self, x):
        
        x = self.maxmap(x)
        rnn_out, _ = self.rnn(x)

        out = self.fc(rnn_out)  # (batch_size, sequence_length, output_size=1)
        out = out.squeeze(-1) # (batch_size, sequence_length)
        return out