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

rng = jax.random.PRNGKey(33)
rng, key = jax.random.split(rng)

normal_samples = jax.random.normal(key, [100_000])

def norm_to_exp(c, lam):
    idx = np.random.randint(0, len(normal_samples), (c.shape[0], 4))
    q = normal_samples[idx]
    return jax.numpy.abs(q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3]) / lam


def norm_to_ray(c, sigma):
    idx = np.random.randint(0, len(normal_samples), (c.shape[0], 2))
    c01 = normal_samples[idx] * sigma 
    return jnp.sqrt(c01[:, 0] ** 2 + c01[:, 1] ** 2)


def norm_to_exp_1d(c, lam):
    idx = np.random.randint(0, len(normal_samples), 4)
    q = normal_samples[idx]
    return jax.numpy.abs(c * q[1] - q[2] * q[3]) / lam


def norm_to_ray_1d(c, sigma):
    idx = np.random.randint(0, len(normal_samples), 2)
    c01 = normal_samples[idx] * sigma 
    c = c * sigma
    return jnp.sqrt(c ** 2 + c01[1] ** 2)

def norm_to_lognorm(a):
    return jnp.exp(a)

def soft_clip(x, hinge_softness=0.3, low=0.0, high=1.0):
    return low + (high - low) * jax.nn.sigmoid(x / hinge_softness)

def phi_add(x, c):
    return (x + c).flatten()

def attack_add(x, b):
    return (x + b).flatten()




def phi_bc(x, c):
    return (x*norm_to_lognorm(c[0]) + c[1]).flatten()

def attack_bc(x, b):
    return (x*b[0] + b[1]).flatten()

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

def sample_rayleigh(key, sz):
    res = jax.random.normal(key, sz)
    res1 = jax.random.normal(key, sz)
    q = jnp.sqrt(res**2 + res1**2)
    return q


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
    J = jax.jacobian(phi1, argnums=1)(x, b, c, type_of_transform) 
    f1 = 0.5*jnp.linalg.slogdet(J.T@J)[1]
    f2 = log_density(c) #User defined
    return -f1 + f2

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
            g.append(jnp.max(bound) * jnp.linalg.norm(beta-b_zero))  # jnp.linalg.norm(beta-b_zero)
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
    """
    Interpolate pseudo xi to obtain 'analytical' xi
    """
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

# ## safe_beta given functions $\xi$,  and points $h(\beta)$, $\beta$ returns if $h$ is certified at $\beta$

def safe_beta(xi, h, hat_g, beta):
    return hat_g(beta) <= -xi(1-h)+xi(0.5)


def construct_gamma(sigma_b=0.4, sigma_c=0.4, sigma_tr=30, sigma_gamma=1.1, sigma_blur=30):
    def _gamma(x, b, c, tr_type:str):
        print(tr_type)
        if tr_type == 'b':  # brightness
            c = c / DEFAULT_SIGMA * sigma_b
            return b+c
        
        if tr_type == 'c':  # contrast
            c = norm_to_lognorm(c / DEFAULT_SIGMA * sigma_c)
            return c * b

        if tr_type == 'cb':
            # contrast then brightness
            c0 = c[0] / DEFAULT_SIGMA * sigma_c
            c1 = c[1] / DEFAULT_SIGMA * sigma_b
            b1 = norm_to_lognorm(c0)*b[0]

            b2 = b[1]*norm_to_lognorm(c0) + c1
            return jnp.array([b1,b2])

        if tr_type == 'gc': ##gamma-contrast
            c0 = c[0] / DEFAULT_SIGMA
            c1 = c[1] / DEFAULT_SIGMA * sigma_c

            c0 = norm_to_ray_1d(c0, sigma_gamma)

            b1 = b[0]*c0
            b2 = b[1]**c0 * norm_to_lognorm(c1)
            return jnp.array([b1, b2])


        if tr_type == 'bt': 
            # brightness translation
            c0 = c[0] / DEFAULT_SIGMA * sigma_b
            c1 = c[1] / DEFAULT_SIGMA * sigma_tr
            c2 = c[2] / DEFAULT_SIGMA * sigma_tr

            b1 = b[0] + c0
            b2 = b[1] + c1 
            b3 = b[2] + c2
            return jnp.array([b1, b2, b3])
        if tr_type == 'cbt':

            c0 = norm_to_lognorm(c[0]*sigma_c)
            c1 = c[1]*sigma_b
            c2 = c[2]*sigma_tr
            c3 = c[3]*sigma_tr



            b0 = c0*b[0]
            b1 = b[1]*c0 +c1
            b2 = b[2] +c2
            b3 = b[3] + c3

            return jnp.array([b0,b1, b2, b3])


        if tr_type == 'tbbc': #translation -  -Blur- Brightness - Contrast
            # Norm(0, 1) -> Laplace(1/sigma_blur) -> Exp(sigma_blur)
            c0 = c[0] / DEFAULT_SIGMA * sigma_tr
            c1 = c[1] / DEFAULT_SIGMA * sigma_tr
            c2 = c[2] / DEFAULT_SIGMA #* sigma_blur
            c3 = c[3] / DEFAULT_SIGMA * sigma_b
            c4 = c[5] / DEFAULT_SIGMA * sigma_c

            x2 = jax.random.normal(key)
            x3 = jax.random.normal(key)
            x4 = jax.random.normal(key)
            c2 = norm_to_exp_1d(c2, sigma_blur)
            b0 = b[0] + c0
            b1 = b[1] + c1
    #         b2 = b[2] + norm_to_exp(c2) * sigma_blur
            b2 = b[2] + c2
            b3 = b[3] + c3 / b[4]
            b4 = norm_to_lognorm(c4)*b[4]

            return jnp.array([b0,b1,b2,b3,b4])

        if tr_type == 'tbbc_ray':

            c0 = c[0] / DEFAULT_SIGMA * sigma_tr
            c1 = c[1] / DEFAULT_SIGMA * sigma_tr
            c2 = c[2] / DEFAULT_SIGMA
            c3 = c[3] / DEFAULT_SIGMA * sigma_b
            c4 = c[5] / DEFAULT_SIGMA * sigma_c



            c2 = norm_to_ray_1d(c2, sigma_blur)

            b0 = b[0] + c0
            b1 = b[1] + c1
            b2 = b[2] + c2
            b3 = b[3] + c3 / b[4]
            b4 = norm_to_lognorm(c4)*b[4]

            return jnp.array([b0,b1,b2,b3,b4])

        if tr_type == 'tr':

            c0 = c[0]*sigma_tr
            c1 = c[1]*sigma_tr


            b0 = b[0] + c0
            b1 = b[1] + c1

            return jnp.array([b0,b1])

        if tr_type == 'ct': 

            c0 = c[0] / DEFAULT_SIGMA * sigma_c
            c1 = c[1] / DEFAULT_SIGMA *sigma_tr
            c2 = c[2] / DEFAULT_SIGMA *sigma_tr


            b0 = b[0] * jnp.exp(c0) #norm_to_lognorm(c0)
            b1 = b[1] + c1
            b2 = b[2] + c2

            return jnp.array([b0,b1,b2])
        if tr_type == "gamma":

            c0 = c[0] / DEFAULT_SIGMA

            c0 = norm_to_ray_1d(c0, sigma_gamma)

            b0 = b[0] * c0
            return jnp.array([b0])


        if tr_type == 'blur_exp': 
            # Norm(0, 1) -> Laplace(1/sigma_blur) -> Exp(sigma_blur)
            b0 = b[0] + norm_to_exp_1d(c[0], sigma_blur)

            return jnp.array([b0])

        if tr_type == 'blur_ray': 
            c0 = c[0] / DEFAULT_SIGMA
            c0 = norm_to_ray_1d(c0, sigma_blur)
            b0 = b[0] + c0
            return jnp.array([b0])
    return _gamma
