import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from layercomponents import PositionalEncoding, PositionWiseFeedForward, MultiHeadAttention
import math

class TransformerEncoderModel(nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, emsize, dropout, ntoken):
        super(TransformerEncoderModel, self).__init__(encoder_layer=encoder_layer, num_layers=num_layers)
        self.model_type = 'Transformer_encoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.input_emb = nn.Embedding(ntoken, emsize)
        self.nfeat = emsize
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        #nn.init.zeros_(self.decoder.bias)
        #nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = src * math.sqrt(self.nfeat)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        return output


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, nfeat, nhead, nhid, nlayers, dropout, device):
        super(TransformerModel, self).__init__(d_model=nfeat, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, device=device, batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nfeat, dropout)
        self.input_emb = nn.Embedding(ntoken, nfeat)  #nodig?
        self.weights = torch.empty(nfeat, ntoken)
        self.nfeat = nfeat
        self.decoder = nn.Linear(nfeat, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.weights, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = src * math.sqrt(self.nfeat)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)[0]
        #print(attn_output.shape)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_mask = None
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        #self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        #self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        #self.pool = torch.func.vmap(torch.mean)

    def generate_mask(self, src, verbose=False):
             
            #src_mask = torch.zeros(len(src), dtype=torch.bool)
            src_mask = src > 1
            #src_mask = src_mask.unsqueeze(1)
            #src_mask = src_mask.unsqueeze(1)
            #src_mask = src & src_mask
            print(src_mask)
            #tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            #seq_length = tgt.size(1)
            #nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            #tgt_mask = tgt_mask & nopeak_mask
            print(src_mask.shape)
            return src_mask

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def pool(self, x):
        p = torch.func.vmap(torch.mean)
        return p(x)

    def forward(self, src, has_mask=True, verbose=False):
        if has_mask:
            device = src.device
            mask = src < 1
            self.src_mask = mask
            #print(self.src_mask.shape)
            #print(src[15])
            #print(self.src_mask[15])
            #while True: continue
            self.src_mask.to(device)
        else:
            self.src_mask = None
        src = src.transpose(0,1) # get data into shape [seqlen, batch_size, d_model]
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.dropout(self.positional_encoding(src))
        
        enc_output = src_embedded
        #print(self.encoder_layers)
        count = 0
        for enc_layer in self.encoder_layers:
            count += 1
            enc_output = enc_layer(enc_output, self.src_mask)    
        enc_output = enc_output.transpose(0,1)
        #print(enc_output[0][0])
        output = torch.zeros(enc_output.size(0), enc_output.size(2))
        #enc_output = enc_output.transpose(0,1)
        #enc_output = enc_output.transpose(1,2)

        #for i in range(len(enc_output)):
            #output[i] = self.pool(enc_output[i])

        for i in range(len(enc_output)):
            output[i] = enc_output[i][0]

        if output.size(0) == 1:
            output = output.squeeze(0)

        return output