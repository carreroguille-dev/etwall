import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.capa_score = nn.Linear(hidden_dim * 2, 1)

    def forward(self, encoder_output, decoder_status):
        decoder_status = decoder_status.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
        
        concat = torch.cat((encoder_output, decoder_status), dim=2)

        scores = self.capa_score(concat).squeeze(-1)

        weights = F.softmax(scores, dim=-1)

        mixed = torch.bmm(weights.unsqueeze(1), encoder_output).squeeze(1)

        return mixed, weights