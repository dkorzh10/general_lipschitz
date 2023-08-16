import torch
import jax
import jax.numpy as jnp
import kornia
import numpy as np


def construct_phi(tr_type, device, sigma_b=0.4, sigma_c=0.4, sigma_tr=30, sigma_gamma=1.1, sigma_blur=30):
    def _phi_bc_torch_batch_and_noise(x):
    
        q = (torch.randn(len(x)) * sigma_c).to(device)
        q1 = (torch.randn(len(x)) * sigma_b).to(device)
#         print(sigma_c)
        q = torch.exp(q[:,None,None,None])
        q1 = q1[:, None, None, None]
        return (x * q + q1)
    
    def _phi_gc_torch_batch_and_noise(x, sigma_g=sigma_gamma, sigma_c=sigma_c):
        q = (torch.randn(len(x)) * sigma_g).to(device)
        q_= (torch.randn(len(x)) * sigma_g).to(device)
        q = torch.sqrt(q**2 + q_**2)

        q1 = (torch.randn(len(x)) * sigma_c).to(device)

        q = q[:, None, None, None]
        q1 = torch.exp(q1[:,None,None,None]) ## Norm to Lornorm
        return (x**q) * q1
    
    def _phi_bt_torch_batch_and_noise(x, sigma=sigma_b, tau=sigma_tr):

#         x = x.to(device)
        q = torch.randn(len(x)) * sigma
        q = q[:,None,None,None]
        x = x + q.to(x.device)

        c1 = (torch.randn(len(x)) * tau).long()*1.0
        c2 = (torch.randn(len(x)) * tau).long()*1.0

        c1 = c1[:, None, None, None]
        c2 = c2[:, None, None, None]

        t = torch.tensor([1,0])
        T = t.expand(len(x), *t.shape).clone()
        T[:, 0] = c1.squeeze().clone()
        T[:, 1] = c2.squeeze().clone()
        out = kornia.geometry.transform.translate(x, T.float().to(x.device), padding_mode='reflection')
        return out

    def _phi_ct_torch_batch_and_noise(x, sigma_c=sigma_c, sigma_tr=sigma_tr):
        contr = (torch.randn(len(x)) * sigma_c)
        contr = contr[:, None, None, None]
        x = contr.to(x.device) * x
#         print("I'm here")
        c1 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        c2 = (torch.randn(len(x)) * sigma_tr).long()*1.0

        t = torch.tensor([1,0])
        T = t.expand(len(x), *t.shape).clone()
        T[:, 0] = c1.squeeze().clone()
        T[:, 1] = c2.squeeze().clone()

        out = kornia.geometry.transform.translate(x.to(device), T.float().to(x.device), padding_mode='reflection')
        return out
    
    def _phi_tr_torch_batch_and_noise(x):
        c1 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        c2 = (torch.randn(len(x)) * sigma_tr).long()*1.0

        t = torch.tensor([1,0])
        T = t.expand(len(x), *t.shape).clone()
        T[:, 0] = c1.squeeze().clone()
        T[:, 1] = c2.squeeze().clone()

        out = kornia.geometry.transform.translate(x, T.float().to(x.device), padding_mode='reflection')
        return out
    
    def _phi_cbt_torch_batch_and_noise(x):
        
        contr = (torch.randn(len(x)) * sigma_c)[:, None, None, None].to(x.device)
        x = contr * x
        b = (torch.randn(len(x)) * sigma_b)[:, None, None, None].to(x.device)
        x = x + b
        c1 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        c2 = (torch.randn(len(x)) * sigma_tr).long()*1.0

        t = torch.tensor([1,0])
        T = t.expand(len(x), *t.shape).clone()
        T[:, 0] = c1.squeeze().clone()
        T[:, 1] = c2.squeeze().clone()

        out = kornia.geometry.transform.translate(x, T.float().to(x.device), padding_mode='reflection')
        return out
    def _phi_gamma_torch_batch_and_noise(x, sigma_g=sigma_gamma):
        q = (torch.randn(len(x)) * sigma_g).to(device)
        q_= (torch.randn(len(x)) * sigma_g).to(device)
        q = torch.sqrt(q**2 + q_**2)

    #     q1 = (torch.randn(len(x)) * sigma_c).to(device)


        q = q[:, None, None, None]
    #     q1 = torch.exp(q1[:,None,None,None]) ## Norm to Lornorm
        return x ** q
    
    
    if tr_type =="cb":
        return  _phi_bc_torch_batch_and_noise
    elif tr_type == "gc":
        return _phi_gc_torch_batch_and_noise
    elif tr_type == "bt":
        return _phi_bt_torch_batch_and_noise
    elif tr_type == "ct":
        return _phi_ct_torch_batch_and_noise
    elif tr_type == "cbt":
        return _phi_cbt_torch_batch_and_noise
    elif tr_type == "tr":
        return _phi_tr_torch_batch_and_noise
    elif tr_type == "gamma":
        return _phi_gamma_torch_batch_and_noise

def attack_cb_torch(x, b):
#     return (x * torch.tensor(b[0].item()) + torch.tensor(b[1].item()))#.flatten()
    return x * b[0]+ b[1]

def attack_gc_torch(x, b):
#     return (x**(torch.tensor(b[0].item()))) * torch.tensor(b[1].item())
    return (x ** b[0]) * b[1]

def attack_bt_torch(x, b):
    x = x + b[0]
    translation = torch.tensor([[b[1], b[2]]]).float().to(x.device) 
    out = kornia.geometry.transform.translate(x, translation, padding_mode='reflection')
    return out

def attack_ct_torch(x, b):
    x = b[0] * x
    translation = torch.tensor([[b[1], b[2]]]).to(x.device)
    out = kornia.geometry.transform.translate(x, translation, padding_mode='reflection')
    return out


def attack_tr_torch(x, b): #translation
    translation = torch.tensor([[b[0], b[1]]]).to(torch.float).to(x.device)
    out = kornia.geometry.transform.translate(x, translation, padding_mode='reflection')
    return out



def attack_cbt_torch(x, b): #trans bright blur contrast
    x = b[0] * x
    x = x + b[1]
    translation = torch.tensor([[b[2], b[3]]]).float.to(x.device) #torch.tensor()
    out = kornia.geometry.transform.translate(x, translation, padding_mode='reflection')
    return out


def attack_gamma_torch(x, b):
#     return x ** (torch.tensor(b[0].item()))
    return x ** b[0]







def safe_beta_tss(tr_type, sigma_c=None, sigma_b=None, sigma_tr=None):
    def _safe_beta_tss_bc(xi, h, beta, sigma=sigma_c, tau=sigma_b):  # sigma=sigma_c, tau=sigma_b
        l = (jnp.log(beta[0]) / sigma) ** 2 + (beta[1]*beta[0] / tau) ** 2
        l = jnp.sqrt(l)
        r = 1/2*(xi(h) - xi(1-h))

        return l<=r
    
    def _safe_beta_tss_bt(xi, h, beta, sigma=sigma_b, tau=sigma_tr):
    
        l1 = (beta[0]/sigma)**2
        l2 = (beta[1]/tau)**2
        l3 = (beta[2]/tau)**2
        l = np.sqrt(l1+l2+l3)
        r = 1/2*(xi(h) - xi(1-h))
        return l <= r
        
    def _safe_beta_tss_tr(xi, h, beta, tau=sigma_tr):
    
        l1 = (beta[0]/tau)**2
        l2 = (beta[1]/tau)**2
        l = np.sqrt(l1+l2)
        r = 1/2*(xi(h) - xi(1-h))

        return l<=r

        return l<=r
    if tr_type == "cb":
        return _safe_beta_tss_bc
    elif tr_type == "bt":
        return _safe_beta_tss_bt
    elif tr_type == "tr":
        return _safe_beta_tss_tr






