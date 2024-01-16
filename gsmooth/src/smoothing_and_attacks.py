import math

import torch
import jax
import jax.numpy as jnp
import kornia
import numpy as np
import cv2


def construct_phi(tr_type, device, sigma_b=0.4, sigma_c=0.4, sigma_tr=30, sigma_gamma=1.1, sigma_blur=30):
    
    def _phi_blur_ray_torch_batch_and_noise(x):
    

        norm1 = torch.randn(len(x))
        norm2 = torch.randn(len(x))
        ray = torch.sqrt(norm1**2 + norm2**2)
        exp = ray ** 2
        blur_radii = sigma_blur * exp

        for i in range(len(x)):
            x[i] = attack_blur_cv2(x[i][None, :], blur_radii[i])
        return x.to(device)

    def _phi_blur_exp_torch_batch_and_noise(x):
    
        blur_radii = norm_to_exp_torch(len(x), sigma_blur)
        for i in range(len(x)):
            x[i] = attack_blur_cv2(x[i][None, :], blur_radii[i])
        return x.to(device)

    def _phi_bc_torch_batch_and_noise(x):
    
        q = (torch.randn(len(x)) * sigma_c).to(device)
        q1 = (torch.randn(len(x)) * sigma_b).to(device)
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
        q = q[:, None, None, None]
        return x ** q
    
    def _phi_tbbc_ray_torch_batch_and_noise(x): 
        x = x.to(device)

        c1 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        c2 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        t = torch.tensor([1,0])
        T = t.expand(len(x), *t.shape).clone()
        T[:, 0] = c1.squeeze().clone()
        T[:, 1] = c2.squeeze().clone()
        x = kornia.geometry.transform.translate(x.to(device), T.float().to(device), padding_mode='reflection')

        x = phi_blur_torch_batch_and_noise(x)

        blur = RayGaussian(sigma_blur)
        x = blur.batch_proc(x)

        b = (torch.randn(len(x)) * sigma_b)[:, None, None, None].to(device)
        x = x + b

        c = torch.exp(torch.randn(len(x)) * sigma_c)[:, None, None, None].to(device)
        x = c * x

        return x
    
    def _phi_tbbc_exp_torch_batch_and_noise(x): 
        x = x.to(device)
        c1 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        c2 = (torch.randn(len(x)) * sigma_tr).long()*1.0
        t = torch.tensor([1,0])
        T = t.expand(len(x), *t.shape).clone()
        T[:, 0] = c1.squeeze().clone()
        T[:, 1] = c2.squeeze().clone()
        x = kornia.geometry.transform.translate(x.to(device), T.float().to(device), padding_mode='reflection')

        x = phi_blur_torch_batch_and_noise(x)
        blur = ExpGaussian(sigma_blur)

        x = blur.batch_proc(x)

        b = (torch.randn(len(x)) * sigma_b)[:, None, None, None].to(device)
        x = x + b

        c = torch.exp(torch.randn(len(x)) * sigma_c)[:, None, None, None].to(device)
        x = c * x

        return x
    
    def _phi_b_torch_batch_and_noise(x):
        q1 = (torch.randn(len(x)) * sigma_b).to(device)
        q1 = q1[:, None, None, None]
        return x + q1
    
    def _phi_c_torch_batch_and_noise(x):
        q = (torch.randn(len(x)) * sigma_c).to(device)
        q = torch.exp(q[:,None,None,None])
        return x * q
    
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
    elif tr_type == "tbbc":
        return _phi_tbbc_torch_batch_and_noise
    elif tr_type =="blur_ray":
        return  _phi_blur_ray_torch_batch_and_noise
    elif tr_type =="blur_exp":
        return  _phi_blur_exp_torch_batch_and_noise
    elif tr_type == "tbbc_ray":
        return _phi_tbbc_ray_torch_batch_and_noise
    elif tr_type == "tbbc_exp":
        return _phi_tbbc_exp_torch_batch_and_noise
    elif tr_type == "b":
        return _phi_b_torch_batch_and_noise
    elif tr_type == "c":
        return _phi_c_torch_batch_and_noise
    else:
        raise Exception("Sorry, invalid transfrom name or it is not added to src")



    



def norm_to_exp_torch(s, lam):
    norm1 = torch.randn(s)
    norm2 = torch.randn(s)
    norm3 = torch.randn(s)
    norm4 = torch.randn(s)
    return torch.abs(norm1 * norm2 - norm3 * norm4) / lam

class Gaussian:
    # it adopts uniform distribution
    def __init__(self, sigma):
        self.sigma = sigma
        self.sigma2 = sigma ** 2.0

    def gen_param(self):
        r = random.uniform(0.0, self.sigma2)
        return r

    def proc(self, input, r2):
        if (abs(r2) < 1e-6):
            return input
        input = input.cpu().numpy()
        out = cv2.GaussianBlur(input.transpose(1, 2, 0), (0, 0), math.sqrt(r2), borderType=cv2.BORDER_REFLECT101)
        if out.ndim == 2:
            out = np.expand_dims(out, 2)
        out = torch.from_numpy(out.transpose(2, 0, 1))
        return out #.cuda()
    
    def proc_new(self, input, r2):
        if (abs(r2) < 1e-6):
            return input
#         print(input.shape)
        input = input.cpu().numpy()
#         print(input.shape)
        out = cv2.GaussianBlur(input.transpose(1, 2, 0), (0, 0), math.sqrt(r2), borderType=cv2.BORDER_REFLECT101)
        if out.ndim == 2:
            out = np.expand_dims(out, 2)
        out = torch.from_numpy(out.transpose(2, 0, 1))
        
        return out #.cuda()

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs
    
class ExpGaussian(Gaussian):
    # it adopts exponential distribution
    # where the sigma is actually lambda in exponential distribution Exp(1/lambda)
    def __init__(self, sigma):
        super(ExpGaussian, self).__init__(sigma)
        self.sigma = sigma

    def gen_param(self):
#         r = - self.sigma * math.log(random.uniform(0.0, 1.0))
        r = np.random.exponential(scale=self.sigma)
        r = r#.to(device)
        return r
    
class RayGaussian(Gaussian):
    # it adopts exponential distribution
    # where the sigma is actually lambda in exponential distribution Exp(1/lambda)
    def __init__(self, sigma):
        super(RayGaussian, self).__init__(sigma)
        self.sigma = sigma

    def gen_param(self):
#         r = - self.sigma * math.log(random.uniform(0.0, 1.0))
        r = np.random.rayleigh(scale=self.sigma)
        return r



    
def attack_blur_cv2(x, b):
    r = b 
    input = x[0].cpu().numpy()
    r = r.item()
    out = cv2.GaussianBlur(input.transpose(1, 2, 0), (0, 0), np.sqrt(r), borderType=cv2.BORDER_REFLECT101)
    out = torch.from_numpy(out.transpose(2, 0, 1))
    return out[None, :].to(x.device)


def attack_cb_torch(x, b):
#     return (x * torch.tensor(b[0].item()) + torch.tensor(b[1].item()))#.flatten()
    return x * b[0]+ b[1]

def attack_b_torch(x, b):
    return x + b[0]

def attack_c_torch(x, b):
    return x * b[0]

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


def attack_tbbc_torch(x, b):  # tr bl br c

    x = x.to(device)
    translation = torch.tensor([[b[0].item(), b[1].item()]]).to(torch.float).to(device) 
    x = kornia.geometry.transform.translate(x, translation, padding_mode='reflection')
    x = attack_blur_cv2(x, b = b[2])
    x = x + torch.tensor(b[3].item())
    x = torch.tensor(b[4].item())*x
    return x





    





# class Gaussian:
#     # it adopts uniform distribution
#     def __init__(self, sigma):
#         self.sigma = sigma
#         self.sigma2 = sigma ** 2.0

#     def gen_param(self):
#         r = np.random.uniform(0.0, self.sigma2)
#         return r

#     def proc(self, input, r2):
#         if (abs(r2) < 1e-6):
#             return input
#         input = input.cpu().numpy()
#         out = cv2.GaussianBlur(input.transpose(1, 2, 0), (0, 0), math.sqrt(r2), borderType=cv2.BORDER_REFLECT101)
#         if out.ndim == 2:
#             out = np.expand_dims(out, 2)
#         out = torch.from_numpy(out.transpose(2, 0, 1))
#         return out #.cuda()
    
#     def proc_new(self, input, r2):
#         if (abs(r2) < 1e-6):
#             return input
# #         print(input.shape)
#         input = input.cpu().numpy()
# #         print(input.shape)
#         out = cv2.GaussianBlur(input.transpose(1, 2, 0), (0, 0), math.sqrt(r2), borderType=cv2.BORDER_REFLECT101)
#         if out.ndim == 2:
#             out = np.expand_dims(out, 2)
#         out = torch.from_numpy(out.transpose(2, 0, 1))
        
#         return out #.cuda()

#     def batch_proc(self, inputs):
#         outs = torch.zeros_like(inputs)
#         for i in range(len(inputs)):
#             outs[i] = self.proc(inputs[i], self.gen_param())
#         return outs
    
# class ExpGaussian(Gaussian):
#     # it adopts exponential distribution
#     # where the sigma is actually lambda in exponential distribution Exp(1/lambda)
#     def __init__(self, sigma):
#         super(ExpGaussian, self).__init__(sigma)
#         self.sigma = sigma

#     def gen_param(self):
# #         r = - self.sigma * math.log(random.uniform(0.0, 1.0))
#         r = np.random.exponential(scale=self.sigma)
#         r = r#.to(device)
#         return r
    
# class RayGaussian(Gaussian):
#     # it adopts exponential distribution
#     # where the sigma is actually lambda in exponential distribution Exp(1/lambda)
#     def __init__(self, sigma):
#         super(RayGaussian, self).__init__(sigma)
#         self.sigma = sigma

#     def gen_param(self):
# #         r = - self.sigma * math.log(random.uniform(0.0, 1.0))
#         r = np.random.rayleigh(scale=self.sigma)
#         return r
    
