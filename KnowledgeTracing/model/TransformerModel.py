import torch
import torch.nn as nn
import copy
from Constant import Constants as C

class AttentionLayer(torch.nn.Module):

    def __init__(self, hidden_size: int, num_heads: int=1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=C.DROP_OUT)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = mask.shape
        assert(mask.shape == (B, L, L))
        assert(input.shape == (L, B, self.hidden_size))

        output, _ = self.attn(
            input, input, input, attn_mask=mask.repeat_interleave(self.num_heads, dim=0))
        output = self.norm(output)
        return input + self.drop(output)


class FCLayer(torch.nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.w2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=C.DROP_OUT)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.w1(input))
        output = self.w2(hidden)
        output = self.norm(output)
        return input + self.drop(output)


class EncodeLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attn = AttentionLayer(hidden_size, num_heads)
        self.fc = FCLayer(hidden_size)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.attn(input, mask))

class DecodeLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attn = AttentionLayer(hidden_size, num_heads)
        self.fc = FCLayer(hidden_size)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.attn(input, mask))

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(torch.nn.Module):
    
    def __init__(self, d_model, num_layers, num_heads, tot_que, tot_concept, tot_ans, max_step, dropout = 0) -> None:
        super().__init__()

        self.embd_que = nn.Embedding(tot_que, embedding_dim = d_model)                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cpt = nn.Embedding(tot_concept, embedding_dim = d_model)
        self.embd_ans = nn.Embedding(tot_ans, embedding_dim = d_model)

        self.encoder = get_clones(EncodeLayer(d_model, num_heads, dropout))
        self.decoder = get_clones(DecodeLayer(d_model, num_heads, dropout))
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        N, B, _ = input.shape

        assert(input.shape == (N, B, self.input_size))

        hidden = self.dense(input)
        for layer in self.layers:
            hidden = layer(hidden, mask)
            # logger.debug("hidden {}".format(hidden))

        output = self.norm(hidden)
        # output = torch.mean(output, dim=0, keepdim=False)
        # output = torch.sum(output, dim=0, keepdim=False)
        return output