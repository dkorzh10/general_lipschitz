import numpy as np
import torch
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.proportion import proportion_confint




def certify(base_classifier,  x, label, Phi, n0, maxn, alpha, batch_size, adaptive=False, device=None, num_classes=None):

    base_classifier.eval()
    x=x.to(device)
    nA, n = 0, 0
    g_phi_list = np.zeros((1,num_classes))
    
    _, counts_selection = sample_noise(base_classifier,x, Phi, n0, device, num_classes)
    cAHat = counts_selection.argmax().item()
            
            
    while n < maxn:
        now_batch = min(batch_size, maxn - n)
        # draw more samples of f(x + epsilon)
        g_phi, counts_estimation = sample_noise(base_classifier, x, Phi, now_batch, device, num_classes)
        
        
        ### TSS things Clopper-Pearson ###
        nA += counts_estimation[cAHat].item()
        n += now_batch
        pABar = lower_confidence_bound(nA, n, alpha)
        ####

        
        g_phi_list = np.concatenate((g_phi_list, g_phi)) 

        g = torch.tensor(g_phi_list[1:])
        g = torch.softmax(g, -1)
        I = torch.mean(g, axis=0)
        V = torch.var(g, axis = 0)/len(g)
        ci = confint(I,V,n=n,alpha=alpha)

        right, idx = torch.sort(ci[1], descending = True)
        right = right#.cpu()
        left = torch.tensor([ci[0][idx[0]]])
        left = left#.cpu()

        if (left[0]>right[1] and adaptive==True):
            break
        else:
            pass

    G_CORRECT_PREDICTION_VIA_I = int(I.argmax().item() == label.item())

    G_CORRECT_PREDICTION_VIA_CI = int(ci[1].argmax().item() == label.item())


    # return left[0], right[1], G_CORRECT_PREDICTION_VIA_I, G_CORRECT_PREDICTION_VIA_CI, pABar, cAHat
    return pABar, cAHat

def sample_noise(base_classifier, x, Phi, num, device, num_classes):
    f_x_eps_list = np.zeros((1,num_classes))
    with torch.no_grad():
        counts = np.zeros(num_classes, dtype=int)
        batch = x.repeat((num, 1, 1, 1))
        batch_noised = Phi(batch).to(device)

        f_x_eps = base_classifier(batch_noised)
        f_x_eps_list = np.concatenate((f_x_eps_list, f_x_eps.cpu().numpy())) 

        predictions = f_x_eps.argmax(1)
        counts +=count_arr(predictions.cpu().numpy(), num_classes)
        
        return f_x_eps_list[1:], counts #counts
    
def count_arr(arr, length) -> np.ndarray:  #might be accelerated probably
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts

def lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
    This function uses the Clopper-Pearson method.
    :param NA: the number of "successes"
    :param N: the number of total draws
    :param alpha: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
    """
#     print(NA, N, alpha, proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0])
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

def confint(I, V, n, alpha, test='Normal'):
    if test == 'T':
        df = n-1
        t = stats.t.ppf(1 - alpha/2, df)
    if test == 'Normal':
        t = stats.norm.ppf(1 - alpha/2)
    
    lower = I - (t * torch.sqrt(V))
    upper = I + (t * torch.sqrt(V))
    #print((upper-lower)*1e15)
    a = torch.stack((lower, upper), dim=0)
    
    return a


def CertAccChecker(safe_beta, betas, hlist, xi, hatg_int):
    for h in hlist:
        flag = True
        for i, beta in enumerate(tqdm(betas)):
            if safe_beta(xi, h, hatg_int, [*beta]).item():
                pass
            else:
                flag = False
                break
        
        if flag==True:
            return h
    return None
    
    
def CertAccCheckerTSS(betas, hlist, xi, safe_beta_tss):
    for h in hlist:
        flag = True
        for i, beta in enumerate(tqdm(betas)):
            if safe_beta_tss(xi, h, [*beta]).item():
                pass
            else:
                flag = False
                break
        
        if flag==True:
            return h
    return None


def pa_isOk_collector(model, loader, Phi, device, n0=100, maxn=2000, alpha=1e-3, batch_size=128, adaptive=False, num_classes=2):
    model.eval()
    model.to(device)
    pas = []
    isOk = []
    
    paTSS =[]
    isOkTSS = []
    with torch.no_grad():
        for k, (image, label) in enumerate(tqdm(loader)):
            # h, pb, _, correct_or_not, pABar, cAHat = certify(model,  image[0], label, Phi, n0,   maxn, alpha, batch_size, adaptive)
            pABar, cAHat = certify(model,  image[0], label, Phi, n0,  maxn, alpha, batch_size, adaptive, device, num_classes)
            # pas.append(h.item())
            # isOk.append(correct_or_not)
            paTSS.append(pABar)
            isOkTSS.append((label==cAHat).int().squeeze())
            
    # pas = np.array(pas)
    # isOk = np.array(isOk)
    paTSS = np.array(paTSS)
    isOkTSS = np.array(isOkTSS)
    

    
    # return pas, isOk, paTSS, isOkTSS
    return paTSS, isOkTSS
        
    
def Accuracy(model, loader, device):
    model.eval()
    isCor = []
    with torch.no_grad():
        model = model.to(device)
        for k, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images) 
            y_pred = y_pred.argmax(dim=-1)
            isCor.append((y_pred == labels).cpu().numpy())
    isCor = np.array(isCor)
    print(isCor.mean())
    return isCor


    

def ERA_Only_ND(model, loader, attack, device, PSN=None, do_transform=False):
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.ones((len(loader)))
        model = model.to(device)
        for k, (images, labels) in enumerate(tqdm(loader)):

                images = images.to(device)
                labels = labels.to(device)
                
                for b in PSN:
                    if do_transform:
                        b = torch.tensor(np.array(b))
                    b = torch.tensor(b)
                    attacked = attack(images, b=b)
#                     attacked = torch.tensor(attacked)
#                     attacked_batch = torch.repeat_interleave(attacked, nsamples, dim=0)
                    labels = labels.to(device)
                    y_pred = torch.nn.Softmax(dim=-1)(model(attacked))#.argmax(dim=-1)

                    y_pred = torch.mean(y_pred, dim=0)
                    y_pred = y_pred.argmax(dim=-1)
                    tensor[k] = tensor[k] * int(y_pred==labels)

                    if not int(y_pred == labels):
                        break

    print(tensor.mean())
    return tensor.mean()



def ERA_Only_For_Smoothed_ND(model, loader, attack, Phi, device, PSN=None, n0=100, maxn=2000, alpha=1e-3, batch_size=256, adaptive=False):
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.ones((len(loader)))
        model = model.to(device)
        for k, (images, labels) in enumerate(tqdm(loader)):

                images = images.to(device)
                labels = labels.to(device)
                
                for b in PSN:
                    b = torch.tensor(np.array(b))
                    attacked = attack(images, b=b)
#                     attacked = torch.tensor(attacked)

                    pABar, cAHat = certify(base_classifier = model,  x=attacked, label=labels, Phi=Phi, n0=n0,  
                         maxn=maxn, alpha=alpha, batch_size=batch_size, adaptive=adaptive, device=device)
                    # left, right, G_CORRECT_PREDICTION_VIA_I, G_CORRECT_PREDICTION_VIA_CI, 

#                     y_pred = torch.nn.Softmax(dim=-1)(model(attacked))#.argmax(dim=-1)
#                     y_pred = torch.mean(y_pred, dim = 0)
#                     y_pred = y_pred.argmax(dim=-1)

#                     y_pred_tss = cAHat
#                     print(G_CORRECT_PREDICTION_VIA_CI)
                    correct = int(cAHat == labels)
                    tensor[k]=tensor[k]* correct #int(G_CORRECT_PREDICTION_VIA_CI)   #int(y_pred == labels) #int(y_pred_tss == labels)

                    if not correct:  #int(y_pred == labels)
                        break
    print(tensor.mean())
    return tensor.mean()