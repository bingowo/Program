import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
from ..Constant import Constants as C


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=2*dim_ff)
        self.layer2 = nn.Linear(in_features=2*dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )
        

class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self , dim_model, heads_en, total_ex ,total_cat, seq_len,dropout):
        super().__init__()
        self.seq_len = seq_len
        self.dim_model = dim_model
        self.embd_ex =   nn.Embedding( total_ex , embedding_dim = dim_model )                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_model )
        self.embd_pos   = nn.Embedding(  seq_len , embedding_dim = dim_model )                  #positional embedding

        self.multi_en = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_en,  )     # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( dim_model )                                            # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, in_ex, in_cat, first_block=True):

        ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_cat = self.embd_cat( in_cat )
            in_cat = torch.mean(in_cat,2)
            #combining the embedings
            out = in_ex + in_cat                      # (b,n,d)

            in_pos = get_pos(self.seq_len)
            in_pos = self.embd_pos( in_pos )
            out = out + in_pos                                      # Applying positional embedding
            out = self.dropout(out)
        else:
            out = in_ex
        


        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape )
        
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1( out )                           # Layer norm
        skip_out = out 
        out, attn_wt = self.multi_en( out , out , out ,
                                attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout(out)
        out = out + skip_out                                    # skip connection

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        out = self.ffn_en( out )
        out = self.dropout(out)
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self,dim_model ,total_in, heads_de,seq_len ,dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_in    = nn.Embedding(  total_in , embedding_dim = dim_model )                  #interaction embedding
        self.embd_pos   = nn.Embedding(  seq_len , embedding_dim = dim_model )                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = Feed_Forward_block( dim_model )                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.layer_norm3 = nn.LayerNorm( dim_model )

        self.dropout = nn.Dropout(p=dropout)

        # self.ff_codes    = nn.Linear(768, 768)
        # self.ff_code    = nn.Linear(768, dim_model)
        # self.ff_tmp    = nn.Linear(dim_model*2, dim_model)

    def forward(self, in_in, in_code,en_out,first_block=True):

         ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            out = self.embd_in(in_in)
            #combining the embedings
            # in_code = self.ff_codes(F.relu(in_code))
            # tmp = self.ff_code(in_code)                 #if using code
            # out = torch.cat((tmp,out),2)
            # out = self.ff_tmp(out)

            in_pos = get_pos(self.seq_len)
            in_pos = self.embd_pos( in_pos )
            out = out + in_pos                                          # Applying positional embedding
            out = self.dropout(out)
        else:
            out = in_in



        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape )
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1( out )
        skip_out = out
        out, attn_wt = self.multi_de1( out , out , out, 
                                     attn_mask=get_mask(seq_len=n)) # attention mask upper triangular
        out = self.dropout(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2( en_out )
        skip_out = out
        out, attn_wt = self.multi_de2( out , en_out , en_out,
                                    attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout(out)
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3( out )                               # Layer norm 
        skip_out = out
        out = self.ffn_en( out )                                    
        out = self.dropout(out)
        out = out + skip_out                                        # skip connection

        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    ##todo add this to device
    return torch.from_numpy( np.triu(np.ones((seq_len ,seq_len)), k=1).astype('bool')).to(C.device)

def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange( seq_len ).unsqueeze(0).to(C.device)

class saint(nn.Module):
    def __init__(self,dim_model,num_en, num_de ,heads_en, total_ex ,total_cat,total_in,heads_de,seq_len,dropout=0):
        super().__init__( )

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones( Encoder_block(dim_model, heads_en , total_ex ,total_cat,seq_len,dropout) , num_en)
        self.decoder = get_clones( Decoder_block(dim_model ,total_in, heads_de,seq_len,dropout)             , num_de)

        self.out = nn.Linear(in_features= dim_model , out_features=1)
    
    def forward(self,in_ex, in_cat,  in_in , in_codes):
        
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x>=1:
                first_block = False
            in_ex = self.encoder[x]( in_ex, in_cat ,first_block=first_block)
            in_cat = in_ex                                  # passing same output as q,k,v to next encoder block

        
        ## pass through each decoder blocks in sequence
        first_block = True
        for x in range(self.num_de):
            if x>=1:
                first_block = False
            in_in = self.decoder[x]( in_in ,in_codes, en_out= in_ex, first_block=first_block )

        ## Output layer
        in_in = self.out( in_in )
        in_in = torch.sigmoid( in_in )
        # in_in = nn.Softmax(dim=2)( in_in )
        return in_in


# ## forward prop on dummy data

# seq_len = 100
# total_ex = 1200
# total_cat = 234
# total_in = 2


# def random_data(bs, seq_len , total_ex, total_cat, total_in = 2):
#     ex = torch.randint( 0 , total_ex ,(bs , seq_len) )
#     cat = torch.randint( 0 , total_cat ,(bs , seq_len) )
#     de = torch.randint( 0 , total_in ,(bs , seq_len) )
#     return ex,cat, de


# in_ex, in_cat, in_de = random_data(64, seq_len , total_ex, total_cat, total_in)


# model = saint(dim_model=128,
#             num_en=6,
#             num_de=6,
#             heads_en=8,
#             heads_de=8,
#             total_ex=total_ex,
#             total_cat=total_cat,
#             total_in=total_in,
#             seq_len=seq_len
#             )

# outs = model(in_ex, in_cat, in_de)

# print(outs.shape)