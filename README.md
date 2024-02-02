# General Lipschitz framework


Algorithms for certification DL models to sematic transformations.

To start, create an image with `nvcr.io/nvidia/pytorch:23.10-py3` container, install required libraries from `requirements.txt`, set suitable path to ImageNet dataset in `datasets.py`, and set up Jupyter lab. In case there is a lack of specific librarym install it additionally. Put in `checkpoints` directory models' weights from [here](https://drive.google.com/file/d/1gQVjx6WBh9PacDJDDdrHjEjM87o_MQEd/view?usp=sharing). Put in `new_results` directory models' weights  from [here](https://drive.google.com/file/d/1P-ukSuRU6cBCeiG1K4ymZsZAEfvwOraU/view?usp=sharing). Put in `tss_weights` directory models' weights  from TTS link [here](https://drive.google.com/file/d/1tW4bTnoxlAFA0KeZGQdHr6Rr9weXJSDS/view?usp=sharing). Don't forget to unzip the downoloaded files.

To obtain presented in the article results (ours, TSS', MP's), go through `notebooks/main_notebook_1.ipynb` and `notebooks/main_notebook_2.ipynb` or run `CUDA_VISIBLE_DEVICES=0 python certify_gl.py configs/config_name.yaml` (don't forget to set GPU in the file). You should choose the perturbation you want to certify the model to in `gamma` (for numericall calculations of required functions), choose parameters of smoothing distributions (scale parameters), create or initialize two function: attack (for ERA) and smoothing phi (for certification and augmentations during training):
For example, for Gamma and Contrast
```python
sigma_c = 0.1
sigma_gamma = 0.1

def gamma(x, b, c, tr_type:str):
    if tr_type == 'gc': ##gamma-contrast
        c0 = c[0] / DEFAULT_SIGMA
        c1 = c[1] / DEFAULT_SIGMA * sigma_c

        c0 = norm_to_ray_1d(c0, sigma_gamma)

        b1 = b[0]*c0
        b2 = b[1]**c0 * norm_to_lognorm(c1)
        return jnp.array([b1, b2])
              
def attack_gc_torch(x, b):
    return (x ** b[0]) * b[1]

# for augmentations
def _phi_gc_torch_batch_and_noise(x, sigma_g=sigma_gamma, sigma_c=sigma_c):
    q = (torch.randn(len(x)) * sigma_g).to(device)
    q_= (torch.randn(len(x)) * sigma_g).to(device)
    q = torch.sqrt(q**2 + q_**2)

    q1 = (torch.randn(len(x)) * sigma_c).to(device)

    q = q[:, None, None, None]
    q1 = torch.exp(q1[:,None,None,None]) ## Norm to Lornorm
    return (x**q) * q1
    
    
```
or

```python
type_of_transform = 'gc'
Phi = construct_phi(type_of_transform, device, sigma_gamma=sigma_gamma, sigma_c=sigma_c)
```
Example of numericall estimations of g(beta)
```python
ns = 10000
x0 = jnp.array([1.1,0.3]) # WHATEVER point
d = 2 ## number of transformation parameters
b_zero = jnp.array([1.0, 1.0]) #identical transformation parameters

betas1 = jnp.linspace(0.4, 2.2, 30) ## set a bit larger range than you want to certify
betas2 = jnp.linspace(0.4, 2.2, 32)
betas = jnp.asarray(list(map(jnp.array, itertools.product(betas1, betas2)))) 

bounds, p, g = compute_normed_bounds(compute_bound, x0, gamma, b_zero, betas, key, ns, d, type_of_transform)

x, xi = pxi_to_xi(p)

z = csaps([betas1, betas2], g.reshape(*betas1.shape, *betas2.shape)) # interpolate


hg = []
for beta in tqdm(betas):
    hat_g = g_to_hat_g(z, beta, b_zero)
    hg.append(hat_g)
hat_g = jnp.asarray(hg)
hatg_int = csaps([betas1, betas2], hat_g.reshape(*betas1.shape, *betas2.shape)) #intterpolation of Integral g(beta) -- required function in certification condition

```
or use prepare function
```python
ns = 20000
b_zero = jnp.array([1.0, 1.0])
x0 = jnp.array([1.1, 1.3]) # Whatever
d = 2

betas1 = jnp.linspace(0.4, 2.2, 30) ## set a bit larger range than you want to certify
betas2 = jnp.linspace(0.4, 2.2, 32)
betas_list = [betas1, betas2]
type_of_transform = 'gc'

Phi = construct_phi(type_of_transform, device, sigma_gamma=sigma_gamma, sigma_c=sigma_c)
attack = attack_gc_torch
res_gc = construct_bounds(ns, b_zero, x0, d, betas_list, type_of_transform)
xi, hatg_int = res_gc
```
and procceed with smoothed classifier predictions collectionusing `pa_isOk_collector` and obtaining minimum value of $h$ in`CertAccChecker`, that is required to accomplish certification conditiom.

For training your own models use `train.py`, e.g.
```
python train.py --run_name cifar10_trans  --dataset cifar10 --arch cifar_resnet110 --type tr --epochs 150  --lr_step_size 50 --batch 512 --device cuda:0 --lr 0.01 --tr 15.0 --lbd 10

python train.py --run_name cifar100_cb  --dataset cifar100 --arch cifar100_resnet110 --type cb --epochs 150  --lr_step_size 50 --batch 512 --device cuda:5 --lr 0.004 --lbd 10 --sigma_c 0.6 --sigma_b 0.6

```

To obtain Gsmooth results, move to `gsmooth/src`, choose or create a proper config,  and run `python gsmooth_certify_resolvable.py configs/CONFIG_NAME.yaml`. Gsmooth CIFAR-10 and CIFAR-100 results are presented in `gsmooth_X.ipynb` notebooks.


In case of problems with cv2,
```
pip uninstall opencv-python
pip uninstall opencv-python-headless

python -m site
```
Delete all cv2, opencv..  dirs from /opt/conda/lib/python3.Y/site-packages

follow the [instructions](https://itsmycode.com/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory/) or from [there](https://github.com/opencv/opencv-python/issues/884):

```
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

apt-get update && apt-get install -y python3-opencv
pip install opencv-python

```
