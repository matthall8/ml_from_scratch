#%%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np

#%%
torch.manual_seed(42)
#%%
class Head(nn.Module):
    def __init__(self, n_embds, n_heads, masked, dropout_rate, device):
        super(Head, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.masked = masked
        self.dropout_rate = dropout_rate
        self.device = device
        head_size = n_embds // n_heads
        self.key_linear_layer = nn.Linear(n_embds, head_size, bias=False)
        self.query_linear_layer = nn.Linear(n_embds, head_size, bias=False)
        self.value_linear_layer = nn.Linear(n_embds, head_size, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, decoder_input, encoder_input=None):
        if encoder_input != None:
            k = self.key_linear_layer(encoder_input)
        else:
            k = self.key_linear_layer(decoder_input)
        q = self.query_linear_layer(decoder_input)
        k_dims = k.shape[2]
        k_transpose = k.permute(0,2,1)
        scores = torch.bmm(q, k_transpose)
        scaled_scores = scores / (k_dims**0.5)
        if self.masked:
            masked_scores = self.apply_attention_mask(scaled_scores)
            softmax_scores = F.softmax(masked_scores, dim=2)
        else: 
            softmax_scores = F.softmax(scaled_scores, dim=2)
        softmax_dropout = self.dropout(softmax_scores)
        if encoder_input != None:
            v = self.value_linear_layer(encoder_input)
        else:
            v = self.value_linear_layer(decoder_input)
        output = torch.bmm(softmax_dropout, v)
        return output
     
    def apply_attention_mask(self, attention_scores):
        # Generate a mask for the lower triangular part of each matrix in the batch
        batch_size = attention_scores.size(0)
        size = attention_scores.size(1)
        mask = torch.tril(torch.ones(batch_size, size, size), diagonal=0).to(self.device)
    
        # Create a tensor of -inf values with the same shape as attention_scores
        negative_inf = torch.full_like(attention_scores, float('-inf')).to(self.device)
    
        # Use torch.where to fill masked positions with -inf
        masked_attention_scores = torch.where(mask.bool(), attention_scores, negative_inf)
    
        return masked_attention_scores
     
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embds, n_heads, masked, dropout_rate,device):
        super(MultiHeadAttention, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.masked = masked
        self.dropout_rate = dropout_rate
        self.device = device
        self.heads = nn.ModuleList([Head(n_embds, n_heads, masked, dropout_rate,device) for _ in range (n_heads)])
        self.proj = nn.Linear(n_embds, n_embds)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, target,source=None):
        if source != None:
            out = torch.cat([h(target,source) for h in self.heads], dim=-1)
        else:
            out = torch.cat([h(target) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class DecoderBlock(nn.Module):
    def __init__(self, n_embds, n_heads,dropout_rate,ff_size, device):
        # n_embds: embedding dimension, n_heads: the number of heads we'd like
        super(DecoderBlock, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.ff_size = ff_size
        self.device = device
        self.masked_sa = MultiHeadAttention(n_embds, n_heads, True, dropout_rate, device)
        self.ca = MultiHeadAttention(n_embds, n_heads, False, dropout_rate, device)
        self.ffwd = FeedForward(n_embds,ff_size,dropout_rate)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)
        self.ln3 = nn.LayerNorm(n_embds)

    def forward(self, source,target):
        masked_multi_head = target + self.masked_sa(self.ln1(target))
        cross_attention = self.ca(target,source)
        target = masked_multi_head + self.ln2(cross_attention)
        target = target + self.ffwd(self.ln3(target))
        return target

class EncoderBlock(nn.Module):
    def __init__(self, n_embds, n_heads,dropout_rate,ff_size, device):
        # n_embds: embedding dimension, n_heads: the number of heads we'd like
        super(EncoderBlock, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.ff_size = ff_size
        self.device = device
        self.sa = MultiHeadAttention(n_embds, n_heads, False, dropout_rate, device)
        self.ffwd = FeedForward(n_embds,ff_size,dropout_rate)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, z):
        z = z + self.sa(self.ln1(z))
        z = z + self.ffwd(self.ln2(z))
        return z


class MatTransformer(nn.Module):
    def __init__(self,max_sequence_len, n_embds, vocab_size, ff_size, dropout_rate, n_heads, n_layers, device):
        super(MatTransformer, self).__init__()
        self.n_embds = n_embds
        self.dropout_rate = dropout_rate
        self.ff_size = ff_size
        self.n_heads = n_heads
        self.device = device
        self.input_embeddings_table = nn.Embedding(vocab_size,n_embds)
        self.output_embeddings_table = nn.Embedding(vocab_size,n_embds)
        self.positional_encodings = self.get_positional_encoding(max_sequence_len, n_embds) 
        self.layer_norm_final = nn.LayerNorm(n_embds)
        self.output_linear_layer = nn.Linear(n_embds, vocab_size)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(n_embds,n_heads,dropout_rate,ff_size,device) for _ in range(n_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_embds,n_heads,dropout_rate,ff_size,device) for _ in range(n_layers)])

    def forward(self, src, target):
        # embeddings and pos encodings
        source_embeddings = self.input_embeddings_table(src).to(self.device)
        pos_encodings = self.positional_encodings[src.shape[1],:].to(self.device)
        source = source_embeddings + pos_encodings
        for blk in self.encoder_blocks: source = blk(source)
        target_embeddings = self.input_embeddings_table(target.long()).to(self.device)
        target = target_embeddings + pos_encodings
        for blk in self.decoder_blocks: target = blk(source, target)

        final_layer_norm =  self.layer_norm_final(target)
        output = self.output_linear_layer(final_layer_norm)

        return output

    def get_positional_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
        return pos_enc

    def num_params(self):
        gpt_params = sum(p.numel() for p in self.parameters())
        emb_params = self.tok_embd.weight.numel()
        print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
        return { 'gpt_params': gpt_params, 'emb_params': emb_params }


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForward, self).__init__()
        self.d_model =d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(p=dropout_rate),     
        )

    def forward(self, x):
        return self.ff(x)
# %%