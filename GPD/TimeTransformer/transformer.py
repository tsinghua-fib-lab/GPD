import torch
import torch.nn as nn
import math
from einops import rearrange
import sys

from TimeTransformer.encoder import Encoder, CrossAttention_Encoder, AdaIN_Encoder
from TimeTransformer.decoder import Decoder
from TimeTransformer.utils import generate_original_PE, generate_regular_PE
import TimeTransformer.causal_convolution_layer as causal_convolution_layer
import torch.nn.functional as F

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



class Transformer1(nn.Module):
    '''
    good transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 spatialloc: list,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 ifkg: bool = True,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        self.ifkg = ifkg
        self.spatialloc = spatialloc
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_mlp = nn.Sequential(
            nn.Linear(self.kgEmb_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timeEmb_mlp = nn.Sequential(
            nn.Linear(self.timeEmb_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timelinear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(d_model, d_model) 
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) 

        
        timeEmb = self.timelinear(timeEmb)
        timeEmb = timeEmb.unsqueeze(1)
        timeEmb = torch.repeat_interleave(timeEmb, self.layernum, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])

        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)
        encoding.add_(timeEmb)
        if self.ifkg:
            kgEmb = self.kgEmb_mlp(kgEmb)
            kgEmb = kgEmb.unsqueeze(1)        
            kgEmb = torch.repeat_interleave(kgEmb, 160, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])
            encoding[:, self.spatialloc[0]:self.spatialloc[1], :].add_(kgEmb)
        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)   # torch.Size([8, 64])  
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        output = self._linear(encoding)

        return output.permute(0,2,1)

   
class Transformer2(nn.Module):
    '''
    Conditions are added to each layer of transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 N: int,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,  
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout) 
            for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_mlp = nn.Sequential(  
            nn.Linear(self.kgEmb_dim, d_model),
        )
        
        self.timeEmb_mlp = nn.Sequential( 
            nn.Linear(self.timeEmb_dim, d_model), 
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) # x shape:  torch.Size([1, 18, 1024])

        kgEmb = self.kgEmb_mlp(kgEmb)
        kgEmb = kgEmb.unsqueeze(1)        
        kgEmb = torch.repeat_interleave(kgEmb, self.layernum, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])
        
        timeEmb = self.timeEmb_mlp(timeEmb)
        timeEmb = timeEmb.unsqueeze(1)
        timeEmb = torch.repeat_interleave(timeEmb, self.layernum, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])

        step = self.step_mlp(t)  
        step = step.unsqueeze(1) 
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)
        
        condition = kgEmb + timeEmb

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding.add_(condition)  # each layer
            encoding = layer(encoding)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
    
class Transformer3(nn.Module):
    '''
    After the conditions are aggregated, they are added to each layer of transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 N: int,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout) 
            for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_linear = nn.Linear(self.kgEmb_dim, d_model)
        
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.forQueryFunc = nn.Sequential(  
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # x.shape:  [64, 64, 265] <- [batchsize, channel, length]
        xEmb = self._embedding(x.permute(0,2,1))                # xEmb.shape [64, 265, 256]

        kgEmb = self.kgEmb_linear(kgEmb)                        # [64, 256]
        kgEmb = kgEmb.unsqueeze(2)                              # [64, 256, 1]
        
        timeEmb = self.timeEmb_linear(timeEmb)                  # [64, 256]
        timeEmb = timeEmb.unsqueeze(2)                          # [64, 256, 1]
        
        kgtEmb = torch.cat((kgEmb, timeEmb), 2)                 # kgtEmb [64, 256, 2]
        
        xQuery = self.forQueryFunc(xEmb)                        # xQuery [64, 265, 256]
        
        # [64, 265, 256] * [64, 256, 2] -> [64, 265, 2]        
        score = torch.bmm(xQuery, kgtEmb)                       # score.shape [64, 265, 2]
        score = F.softmax(score, dim = 2)
        
        # [64, 265, 2] * [64, 256, 2] -> [64, 265, 256]
        condition = torch.bmm(score, torch.transpose(kgtEmb, 1, 2))  # condition: [64, 265, 256]  


        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # prepare embedding into encoder
        encoding = xEmb
        encoding = encoding + step_emb
        
        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)   # torch.Size([8, 64]) 
            encoding.add_(positional_encoding)  

        # Encoder stack
        for layer in self.layers_encoding:
            encoding = encoding + condition  
            encoding = layer(encoding)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
      
class Transformer4(nn.Module):
    '''
        cross attention
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,  
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([CrossAttention_Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_linear = nn.Linear(self.kgEmb_dim, d_model)
        
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) # x shape:  torch.Size([1, 18, 1024])

        kgEmb = self.kgEmb_linear(kgEmb)                        # [64, 256]
        kgEmb = kgEmb.unsqueeze(2)                              # [64, 256, 1]
        
        timeEmb = self.timeEmb_linear(timeEmb)                  # [64, 256]
        timeEmb = timeEmb.unsqueeze(2)                          # [64, 256, 1]
        
        kgtEmb = torch.cat((kgEmb, timeEmb), 2)                 # kgtEmb [64, 256, 2]

        step = self.step_mlp(t)  
        step = step.unsqueeze(1) 
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)   
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, kgtEmb)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
    
class Transformer5(nn.Module):
    '''
        Adaptive LayerNorm
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([AdaIN_Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_linear = nn.Linear(self.kgEmb_dim, d_model)
        
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) # x shape:  torch.Size([1, 18, 1024])
        
        kgtEmb = torch.cat((kgEmb, timeEmb), 1)            

        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, kgtEmb)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)    


