import torch

# def phi_bc_torch_batch_and_noise(x):
#     q = (torch.randn(len(x))*sigma_c).to(device)
#     q1 = (torch.randn(len(x)) * sigma_b).to(device)
    
#     q = torch.exp(q[:,None,None,None])
#     q1 = q1[:, None, None, None]
#     return (x * q + q1)

# def attack_bc_torch(x, b):
#     return (x*torch.tensor(b[0].item()) + torch.tensor(b[1].item()))#.flatten()