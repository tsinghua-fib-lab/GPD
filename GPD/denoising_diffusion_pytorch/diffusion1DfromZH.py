import torch
from torch import nn
import sys

class SimpleDiffusion(nn.Module):
    def __init__(
            self,
            device,
            denoisingModel,
            num_steps = 100,
            seqlength = 1,
            channels = 2,
    ):
        super().__init__()
        self.device = device
        self.model = denoisingModel
        self.num_steps = num_steps    
        betas = torch.linspace(-6, 6, num_steps)
        self.betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5  
        self.alphas = 1-self.betas			
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)  
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.seqlength = seqlength
        self.channels = channels

        assert self.alphas.shape == self.alphas_prod.shape \
            == self.alphas_prod_p.shape == self.alphas_bar_sqrt.shape \
                == self.one_minus_alphas_bar_log.shape \
                    == self.one_minus_alphas_bar_sqrt.shape
        print("all the same shape", self.betas.shape)	


    def q_x(self, x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise) 

    def diffusion_loss_fn(self, x_0, emb):
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.num_steps, size=(batch_size // 2,))		
        t = torch.cat([t, self.num_steps - 1 - t], dim=0)					 
        t = t.unsqueeze(-1).to(self.device)

 
        a = self.alphas_bar_sqrt[t]
        a = a.unsqueeze(2).to(self.device)
        if self.seqlength>1:
            a = torch.repeat_interleave(a, self.seqlength, dim=2)

        aml = self.one_minus_alphas_bar_sqrt[t]
        aml = aml.unsqueeze(2).to(self.device)
        if self.seqlength>1:
            aml = torch.repeat_interleave(aml, self.seqlength, dim=2)
        e = torch.randn_like(x_0).to(self.device)
        x = x_0 * a + e * aml
        output = self.model(x, t.squeeze(-1), emb, None)

        return (e - output).square().mean()

    def p_sample_loop_all(self, shape, emb):
        cur_x = torch.randn(shape).to(self.device)
        x_seq = [cur_x]
        for i in reversed(range(self.num_steps)):
            cur_x = self.p_sample(cur_x, i, emb)
            x_seq.append(cur_x)
        return x_seq


    def p_sample(self, x, t, emb):
        t = torch.tensor([t]).to(self.device)

        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        coeff = coeff.to(self.device)

        eps_theta = self.model(x, t, emb, None)
        aaa = (1 / (1 - self.betas[t]).sqrt()).to(self.device)

        mean = aaa * (x - (coeff * eps_theta))

        z = torch.randn_like(x).to(self.device)
        sigma_t = self.betas[t].sqrt()
        sigma_t = sigma_t.to(self.device)

        sample = mean + sigma_t * z

        return (sample)
    
    def forward(self, x, emb):
        return self.diffusion_loss_fn(x, emb)
    
    def sample(self, emb=None, batch_size=10000):
        shape = [batch_size, self.channels, self.seqlength]
        x = torch.randn(shape).to(self.device)
        for i in reversed(range(self.num_steps)):
            x = self.p_sample(x, i, emb)
        return x
    
