from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, n_layers=2, dropout=0.5):
        super().__init__()

        self.embedding = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_size = self.embedding.embedding_dim
        
        self.lstm = nn.LSTM(self.input_size, hidden_size, n_layers, 
                            dropout=self.dropout, bidirectional=True)

        
    def forward(self, sequences, sequence_lengths, hidden=None):
        """
        sequences: LongTensors of variable length, with padding enabled.

        """

        embedded = self.embedding(sequences)
        packed = pack_padded_sequence(embedded, sequence_lengths)
        
        outputs, (hidden, cell_state) = self.lstm(packed, None)
        outputs, output_lengths = pad_packed_sequence(outputs)

        # Combine bidirectional outputs
        h = self.hidden_size
        outputs = outputs[:, :, :h] + outputs[:, :,  h:]

        return outputs, hidden
