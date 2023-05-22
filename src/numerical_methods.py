import jax
import jax.numpy as jnp
import numpy as np
import scipy
import itertools
import equinox
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from csaps import csaps

DEFAULT_SIGMA = 1.0

# def norm_to_exp(a):
#     return jnp.log(2/jax.lax.erfc(a/jnp.sqrt(2)))

# def norm_to_exp(a):
#     lam = 1
# #     q = jnp.sqrt(2 * DEFAULT_SIGMA ** 2 * [-1/2 * jnp.log(2 * jnp.pi * DEFAULT_SIGMA ** 2) - a])
# # jnp.exp(-q/ 30) / 30
#     h = 0.5 * (1 + jax.lax.erf(a / jnp.sqrt(2) / DEFAULT_SIGMA))
#     return - lam * jnp.log(1 - h)

# def norm_to_exp1(a, lam):
#     h = 0.5 * (1 + jax.lax.erf(a / jnp.sqrt(2) / DEFAULT_SIGMA))
#     return - lam * jnp.log(1 - h)

# def norm_to_ray(a):
#     return jnp.sqrt(-jnp.log((1-0.5*jax.lax.erfc(-a/jnp.sqrt(2)))**2))    


def norm_to_lognorm(a):
    return jnp.exp(a)

def soft_clip(x, hinge_softness=0.3, low=0.0, high=1.0):
    return low + (high - low) * jax.nn.sigmoid(x / hinge_softness)

def phi_add(x, c):
    return (x + c).flatten()

def attack_add(x, b):
    return (x + b).flatten()




def phi_bc(x, c):
    #return x*jnp.exp(c[0]) + c[1]
    #return  (x*norm_to_ray(c[0]) + c[1])
    return (x*norm_to_lognorm(c[0]) + c[1]).flatten()
    #return x**(norm_to_lognorm(c[0])) + c[1]
    #return x*c[0] + c[1]
def attack_bc(x, b):
    return (x*b[0] + b[1]).flatten()
    #return soft_clip(x*b[0] + b[1])
    #return x**b[0] + b[1]
def phi_gamma(x, c):
    return (x**(norm_to_exp(c[0]))).flatten()

def attack_gamma(x, b):
    return (x**(b[0])).flatten()


def phi_gamma_b(x, c):
    return x**(norm_to_lognorm(c[0])) + norm_to_exp(c[1])

def attack_gamma_b(x, b):
    return x**(b[0]) + b[1]

def sample_normal(key, sz):
    return jax.random.normal(key, sz)

def sample_lognormal(key, sz):
    res = jax.random.normal(key, sz)
    return jnp.exp(res)

# def sample_rayleigh(key, sz):
#     res = jax.random.normal(key, sz)
#     return norm_to_ray(res)

def sample_rayleigh(key, sz):
    res = jax.random.normal(key, sz)
    res1 = jax.random.normal(key, sz)
    q = jnp.sqrt(res**2 + res1**2)
    return q

# def sample_exponential(key, sz):
#     res = jax.random.normal(key, sz)
#     return norm_to_exp(res)

def logpdf_normal(a):
    return jax.scipy.stats.norm.logpdf(a)

def logpdf_lognormal(a):
    return -0.5*jnp.log(a)**2 - 0.5*jnp.log(2*jnp.pi)

def logpdf_rayleigh(a):
    return jnp.log(a) - a**2/2

def logpdf_exponential(a):
    return -a


def build_phi1(attack, phi):
    def phi1(x,b,c):
        return phi(attack(x,b), c)
    return phi1


def log_density(c):
    """
        The logarithm of the density that is used for smoothing
    """
    #return jax.scipy.stats.norm.logpdf(c[0]) + jax.scipy.stats.norm.logpdf(c[1])
    return sum(jax.scipy.stats.norm.logpdf(z) for z in c)


def compute_log_rho(x, phi1, b, c, type_of_transform):
    """Computes the log-density at the point
        y = phi(attack(x, b), c), where c is sampled from the density above
    """
    #x1 = attack(x, b)
    J = jax.jacobian(phi1, argnums=1)(x, b, c, type_of_transform) 
    f1 = 0.5*jnp.linalg.slogdet(J.T@J)[1]
    f2 = log_density(c) #User defined
    return f1 - f2

def compute_dcdb(x, phi1, b, c, type_of_transform): 
    """Computes the derivative of the smoothing parameter with respect to b given x.
       can be reduced to the solution of a least squares problem
    """
    
    J = jax.jacobian(phi1, argnums=2)(x, b, c, type_of_transform) 
    rhs = jax.jacobian(phi1, argnums=1)(x, b, c, type_of_transform)
    cf = jnp.linalg.pinv(J)@rhs #dc/db !
    return -cf


@equinox.filter_jit()
def grad_log_rho(x, phi1, b, c, type_of_transform):
    """Computes the gradient of the logarithm of the density with respect to the attack parameter
    """
    g1 = jax.jacobian(compute_log_rho, argnums=2)(x, phi1, b, c, type_of_transform) #Maybe these two calls can be combined

    g2 = jax.jacobian(compute_log_rho, argnums=3)(x, phi1, b, c, type_of_transform)
    return g1 + g2@compute_dcdb(x, phi1, b, c, type_of_transform)

grad_log_rho_vect = equinox.filter_jit((jax.vmap(grad_log_rho, in_axes=(None, None, None, 0, None))))


# #Additional: Laplace density formula.
# def compute_log_rho_laplace(y, x, b, c, s):
#     x1 = attack(x, b)
#     J = jax.jacobian(phi, argnums=1)(x1, c)
#     M = J.T@J + (s**2)*jnp.eye(J.shape[1])
#     mu = (y-phi(x1, c)) + J@c
#     am = jnp.linalg.solve(M, J.T@mu) 
#     log_rho = -0.5*jnp.linalg.slogdet(M)[1] - 1/(2*s**2)*(jnp.dot(mu, mu) \
#                 - jnp.dot(M@am, am)) - 0.5*jnp.log(2*jnp.pi) #The last term depends! on the dimension
#     return log_rho





def compute_bound(x0, phi1, b_zero, b, c, type_of_transform):
    eta = grad_log_rho_vect(x0, phi1, b, c, type_of_transform)
    t = (b - b_zero) / jnp.linalg.norm(b-b_zero)
    eta = eta@(t)
    eta = eta - eta.mean()
    #eta = -c[:, 0] + c[:, 1]*jnp.exp(c[:, 0])*b[1]
    bound = jnp.cumsum(jnp.sort(eta)[::-1])/eta.shape[0] 

    return bound




def compute_normed_bounds(bound_fn, x, phi1, b_zero, betas, key, ns, d, type_of_transform, sigma=DEFAULT_SIGMA, eps=1e-3):
    '''
    normed bound (over h and beta) and max_h (bound(beta, h))
    '''
    bounds = []
    g = []
    for beta in tqdm(betas):

        key, key1 = jax.random.split(key)
        c = jax.random.normal(key1, [ns, d])*sigma
        
        if jnp.linalg.norm(beta-b_zero) > eps:
            bound = bound_fn(x, phi1, b_zero, beta, c, type_of_transform)
            if jnp.max(bound > 0.0):
                bounds.append(bound / jnp.max(bound))
            else:
                bounds.append(jnp.zeros_like(bound))
            g.append(jnp.max(bound) * jnp.linalg.norm(beta-b_zero))
        else:
            bound = bound_fn(x, phi1, b_zero, beta, c, type_of_transform)
            g.append(0.0)
            bounds.append(jnp.zeros_like(bound))
    bounds = jnp.asarray(bounds)
    print(bounds.shape)
    g = jnp.asarray(g)
    print(g.shape)
    p = jnp.max(bounds, axis=0)
    print(p.shape)
    return bounds, p, g
    
    
    
    
def pseudo_xi(norm_bound):
    N = len(norm_bound)
    x_axis = np.linspace(0,1,N)
    cumsum = jax.numpy.cumsum(1/norm_bound) / N
    xi = jax.numpy.interp(x_axis,x_axis, cumsum)
    return x_axis, xi


def pxi_to_xi(norm_bound):
    x_axis, pxi = pseudo_xi(norm_bound)
    pxi_int = scipy.interpolate.interp1d(x_axis, pxi)
    return x_axis,pxi_int



# @equinox.filter_jit()

class gt:
    def __init__(self, f, a,b):
        self.f = f
        self.a = a
        self.b = b
    def __call__(self, t):
        z = (1-t)*self.a + t*self.b
        return self.f(z.tolist())

    
def g_to_hat_g(g_interpolated, beta, bzero):
    
    '''
    takes interpolated function g(beta) 
    and returns inegrated function hat{g}(beta) = int_beta0^beta g(beta)dbeta
    '''
    g = g_interpolated  # z
    a = bzero
    b = beta

    gt_f = gt(g, a, b)
    return scipy.integrate.quad(func=gt_f, epsabs=1e-4, epsrel=1e-3, a=0.0, b=1.0)[0]

### safe_beta given functions $\xi$,  and points $h(\beta)$, $\beta$ returns if $h$ is certified at $\beta$

def safe_beta(xi, h, hat_g, beta):
    return hat_g(beta) <= -xi(1-h)+xi(0.5)

