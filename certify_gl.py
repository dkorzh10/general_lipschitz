import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
import yaml

import click
import scipy
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
# from torchvision.models.resnet import resnet50

from architectures import get_architecture
from datasets_utils import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from src.numerical_methods import *
from src.certification_utils import *
from src.smoothing_and_attacks import *
from src.utils import *


def make_test_dataset(config):
    test_dataset = get_dataset(config["dataset"], "test")
    pin_memory = (config["dataset"] == "imagenet")
    np.random.seed(42)
    idxes = np.random.choice(len(test_dataset), config["NUM_IMAGES_FOR_TEST"], replace=False)
    
    ourdataset = make_our_dataset_v2(test_dataset, idxes)
    ourdataloader = DataLoader(ourdataset, shuffle=False, batch_size=1,
                         num_workers=6, pin_memory=False)
    return ourdataset, ourdataloader


def load_model(config):
    device = torch.device(config["device"])
    model = get_architecture(arch=config["arch"], dataset=config["dataset"], device=device)
    checkpoint = torch.load(config["base_classifier"], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    return model


def construct_bounds(ns, b_zero, x0, d, betas_list, type_of_transform, sigmas=[1, 1, 1, 1, 1]):
    shape = [b.shape[0] for b in betas_list]
    print("I'm here")
    shape = tuple(shape)
    betas = jnp.asarray(list(map(jnp.array, itertools.product(*betas_list))))
    sigma_b, sigma_c, sigma_tr, sigma_gamma, sigma_blur = sigmas
    gamma = construct_gamma(sigma_b=sigma_b, sigma_c=sigma_c, sigma_tr=sigma_tr, sigma_gamma=sigma_gamma, sigma_blur=sigma_blur)
    bounds, p, g = compute_normed_bounds(compute_bound, x0, gamma, b_zero, betas, key, ns, d, type_of_transform)
    x, xi = pxi_to_xi(p)
    z = csaps(betas_list, g.reshape(shape))
    
    hg = []

    for beta in tqdm(betas):
        hat_g = g_to_hat_g(z, beta, b_zero)
        hg.append(hat_g)

    hat_g = jnp.asarray(hg)

    hatg_int = csaps(betas_list, hat_g.reshape(shape)) #
    return xi, hatg_int


def calculate_general(config):
    
    # defining and loading main parameters and fucntions
    xi_tss = scipy.stats.norm.ppf
    
    device = torch.device(config["device"])
    sigmas = config["sigmas"]
    device = torch.device("cuda")

    n0 = config["n0"]
    maxn = config["maxn"]
    adaptive = config["adaptive"]
    alpha = config["alpha"]
    bs = config["batch"]
    num_classes = config["num_classes"]
    
    
    b_zero = jnp.array([1., 0.])
    x0 = jnp.array([1.1, 1.3]) # Whatever
    d = 2
    
    # loading base classifier f
    model = load_model(config)
    
    # creating test dataset to certify on
    ourdataset, ourdataloader = make_test_dataset(config)
    
    # constructing smoothing transfrom phi
    phi = construct_phi(config["transform"], device=device, **sigmas)
    
    
    # constructiong xi and hatg
    b_zero = jnp.array(config["b_zero"])
    x0 = jnp.array(config["x0"])
    d = config["dimenshion"]
    type_of_transform = config["transfrom"]
    
    betas_dict = config["betas_estimation"]
    betas_list = []
    for key in betas_dict:
        beta = betas_dict[key]
        betas_list.append(jnp.linspace(beta["left"], beta["right"], beta["db"]))
        
    
    xi, hatg_int = construct_bounds(ns, b_zero, x0, d, betas_list, type_of_transform)
    
    # defining and loading attack
    attack = construct_attack(type_of_transform)
    
    # calculating benign (vanilla) accuracy of base classifier f
    # Accuracy of non-smoothed model on original images
    benign_acc = Accuracy(model, loader=ourdataloader, device=device)
    print(f"Benign accuracy {benign_acc}")
    
    # creating attack set B to certify model on
    betas_attack_dict = config["betas_certification"]
    betas_attack_list = []
    for key in betas_attack_dict:
        beta_attack = betas_attack_dict[key]
        betas_atttack_list.append(np.linspace(beta_attack["left"], beta_attack["right"], beta_attack["db"]))
    betas_attack = np.asarray(list(map(np.array, itertools.product(*betas_attack_list))))

    # calculating statistics
    paCP, isOkCP = pa_isOk_collector(model, loader=ourdataloader, Phi=Phi, device=device,
                           n0=n0, maxn=maxn, alpha=alpha, batch_size=bs, adaptive=adaptive,
                             num_classes=num_classes)
    h_acc = np.mean(isOkCP)
    print(f"Ordinary accuracy of Smoothed Classiifer {h_acc}")
    
    
    # calculate empirically robust accuracy of f
    f_era = None
    if config["calculate_f_era"]:
        f_era = ERA_Only_ND(model, ourdataloader, attack=attack, device=device, PSN=betas_attack)
        print(f"f_era {f_era}")
    

    # calculate empirically robust accuracy of h (might be very time consuming)
    h_era = None
    if config["calculate_f_era"]:
        h_era = ERA_Only_For_Smoothed_ND(model, ourdataloader, attack, Phi, device, 
                                 PSN=betas_attack, n0=n0, maxn=maxn, alpha=alpha, 
                                 batch_size=bs, adaptive=adaptive, num_classes=num_classes)
        print(f"h_era {h_era}")
        

    # calulate our certified robust accuracy (CRA)
    hlist = np.linspace(config["hlist"]["left"], config["hlist"]["right"], config["hlist"]["n_steps"])
    hmin_ours = CertAccChecker(safe_beta, betas=betas_attack, hlist=hlist, xi=xi, hatg_int=hatg_int)

    if hmin_ours:
        cert_acc_ours = ((paCP > hmin_ours).astype("int") * isOkCP).mean()
    else:
        cert_acc_ours = 0
        hmin_ours = None
    print(f"Cert Acc {type_of_transform} ours is {cert_acc_ours}.  h_min is {hmin_ours}")
    

    # calculate TSS' CRA if applicable
    sb_tss = safe_beta_tss(type_of_transform, **sigmas)
    hmin_tss = CertAccCheckerTSS(betas=betas_attack, hlist=hlist, xi=xi_tss, safe_beta_tss=sb_tss)
    if hmin_tss:
        cert_acc_tss = ((paCP > hmin_tss * isOkCP).mean()
    else:
        cert_acc_tss = 0
        hmin_tss = None
    print(f"Cert Acc {type_of_transform} ours is {cert_acc_tss}.  h_min is {hmin_tss}")
    
    
    # calculate MP's CRA if applicable
    
    

    
@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config["b_zero"])
    
#     if config["gpu"]:
#         os.environ['CUDA_VISIBLE_DEVICES']=str(config["gpu"])
#         os.environ['CUDA_LAUNCH_BLOCKING']="1"
        


    calculate_general(config)

if __name__ == "__main__":
    main()
