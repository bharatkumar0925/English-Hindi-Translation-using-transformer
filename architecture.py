import random
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
#        self.multi_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.hidden_dim = hidden_dim

    def forward(self, src):
        embedding = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedding)
#        attn_output, attn_weights = self.multi_attention(outputs, outputs, outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.layers(outputs)
        # Return all outputs for each time step as well as hidden, cell states
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, output_dim)
#            nn.LeakyReLU()
        )
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=16, batch_first=True, dropout=0.1)

    def forward(self, target, hidden, cell, encoder_outputs):
        # target: (batch_size)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell: (num_layers, batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)

        target = target.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(target)  # (batch_size, 1, embed_dim)

        # Prepare query, key, value for MHA
        # Query should be (batch_size, query_len=1, hidden_dim)
        query = hidden[-1].unsqueeze(1)  # (batch_size, 1, hidden_dim)
        key = encoder_outputs            # (batch_size, src_len, hidden_dim)
        value = encoder_outputs          # (batch_size, src_len, hidden_dim)

        # Apply MultiHeadAttention
        context, attn_weights = self.multihead_attn(query, key, value)  # context: (batch_size, 1, hidden_dim)

        # Concatenate embedded target and context
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_dim + hidden_dim)

        # Pass through RNN
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Generate prediction
        prediction = self.fc(torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1))  # (batch_size, output_dim)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trz, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trz_len = trz.shape[1]
        trz_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trz_len, trz_vocab_size).to(device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trz[:, 0]  # <START> token

        for t in range(1, trz_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            # Get the predicted token
            top1 = output.argmax(1)

            # Use teacher forcing
            teacher_forcing = random.random() < teacher_forcing_ratio
            input = trz[:, t] if teacher_forcing else top1

        return outputs
