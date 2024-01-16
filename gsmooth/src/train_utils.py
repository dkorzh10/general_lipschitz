import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# noise generator : shape, type
def sample_noise(shape, sigma, type):

    if type =='gaussian':
        noise = sigma*torch.randn(shape)
    elif type == 'exp':
        noise = torch.from_numpy(np.random.exponential(scale=sigma, size=shape)).float()
    elif type == 'uniform':
        noise = sigma * torch.rand(shape)
    elif type == 'folded_gaussian':
        noise = sigma * torch.abs(torch.randn(shape))
    else:
        raise NotImplementedError
    return noise



def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()
    print('Logs saved in '+filename)

def log(filename: str, text: str,prt=False):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()
    if prt:
        print(text)






'''find functions and classes without comment, if there is comment, might change the result!!'''


def find_funcs(file, func):
    with open(file, 'r') as fp:
        code = fp.read()
        start = code.find('def ' + str(func))

        # end with another class or function or the end of file
        end1 = code[start + 3:].find('class')
        end2 = code[start + 3:].find('def')

        end = min(end1, end2)
        if end==-1:
            return code[start:]+'\n'
        else:
            return code[start:start + end] + '\n'


