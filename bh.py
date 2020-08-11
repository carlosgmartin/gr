import numpy as np
from sympy import symbols, sin, pprint, Array
from sympy.diffgeom import (
    Manifold,
    Patch,
    CoordSystem,
    metric_to_Christoffel_2nd,
    TensorProduct as TP,
    twoform_to_matrix
)
from sympy.utilities.lambdify import lambdify
from timeit import default_timer
from itertools import islice
import matplotlib.pyplot as plt
from pdb import set_trace

manifold = Manifold(None, 4)
patch = Patch(None, manifold)
m, q = symbols('m, q')

if True:
    coord = CoordSystem(None, patch, ['u', 'r', 'θ', 'φ'])
    u, r, θ, φ = coord.coord_functions()
    du, dr, dθ, dφ = coord.base_oneforms()
    metric = -(1 - m/r + (q/r)**2) * TP(du, du) + TP(du, dr) + TP(dr, du) + r**2 * (TP(dθ, dθ) + sin(θ)**2 * TP(dφ, dφ))
else:
    # The Kerr spacetime: A brief introduction by Matt Visser (Equation 42)
    # https://arxiv.org/pdf/0706.0622.pdf#page=11
    coord = CoordSystem(None, patch, ['t', 'r', 'θ', 'φ'])
    t, r, θ, φ = coord.coord_functions()
    dt, dr, dθ, dφ = coord.base_oneforms()
    metric = -TP(dt, dt) + TP(dr, dr) + r**2 * (TP(dθ, dθ) + sin(θ)**2 * TP(dφ, dφ)) + (m/r - (q/r)**2) * TP(dt + dr, dt + dr)

metric_array = Array(twoform_to_matrix(metric))
christoffel_array = metric_to_Christoffel_2nd(metric)
christoffel_f = lambdify(coord.coord_functions() + [m, q], christoffel_array)
metric_f = lambdify(coord.coord_functions() + [m, q], metric_array)

def to_array(a, shape):
    a = np.array(a, dtype=object)
    b = np.empty(shape)
    for index in np.ndindex(a.shape):
        b[index] = a[index]
    return b

def Γ(x, *args):
    return to_array(christoffel_f(*x, *args), christoffel_array.shape + x.shape[1:])

def Γ(x, m, q):
    # fast version
    u, r, θ, φ = x
    cos_θ = np.cos(θ)
    sin_θ = np.sin(θ)
    y = np.zeros(christoffel_array.shape + x.shape[1:])

    y[0, 0, 0] = m / (2 * r**2) - q**2 / r**3
    y[0, 2, 2] = -r
    y[0, 3, 3] = -r * sin_θ**2

    y[1, 0, 0] = y[0, 0, 0] * (r**2 - r * m + q**2) / r**2
    y[1, 1, 0] = -y[0, 0, 0]
    y[1, 0, 1] = y[1, 1, 0]
    y[1, 2, 2] = -(r**2 - r * m + q**2) / r
    y[1, 3, 3] = y[1, 2, 2] * sin_θ**2

    y[2, 2, 1] = 1/r
    y[2, 1, 2] = y[2, 2, 1]
    y[2, 3, 3] = -sin_θ * cos_θ

    y[3, 3, 1] = 1/r
    y[3, 1, 3] = y[3, 3, 1]
    y[3, 3, 2] = cos_θ / sin_θ
    y[3, 2, 3] = y[3, 3, 2]

    return y

def g(x, *args):
    return to_array(metric_f(*x, *args), metric_array.shape + x.shape[1:])

def integrate(x, v, Γ, h, *args):
    while True:
        a = -(Γ(x, *args) * v[None, :, None] * v[None, None, :]).sum((1, 2))
        yield x
        v = v + a * h
        x = x + v * h

m = 1
q = .4

inner_horizon = 1/2 * (m - np.sqrt(m**2 - 4 * q**2))
outer_horizon = 1/2 * (m + np.sqrt(m**2 - 4 * q**2))
photon_sphere = 1/4 * (3 * m + np.sqrt(9 * m**2 - 32 * q**2))

# innermost circular orbit (for uncharged particles)
# https://journals-aps-org.proxy.library.cmu.edu/prd/pdf/10.1103/PhysRevD.83.024021
# isco = np.roots([-8*q**4, 9*q**2*m, -3*m**2, m][::-1])[0]

ds2 = 0 # 0 for lightlike, 1 for spacelike, -1 for timelike

if True:
    # spiral
    def dθ_r(r, m, q, ds2):
        return np.sqrt(r**2 * (1 + ds2) - m * r + q**2) / r**2

    r = np.linspace(0, 2, 100 + 1) * (photon_sphere - outer_horizon) + outer_horizon

    x = np.empty((4, len(r)))
    x[0] = 0
    x[1] = r
    x[2] = np.pi/2
    x[3] = 0

    v = np.empty_like(x)
    v[0] = 1
    v[1] = 0
    v[2] = dθ_r(r, m, q, ds2)
    v[3] = 0
else:
    # burst
    def du_a(a, r, m, q, ds2):
        return r**2 * (np.cos(a) + np.sqrt(np.cos(a)**2 + (q**2 - m * r + r**2) * (np.sin(a)**2 - ds2) / (r**2))) / (q**2 + r * (r - m))

    a = np.linspace(0, 2 * np.pi, 50 + 1) + np.pi/2
    r = photon_sphere

    x = np.empty((4, len(a)))
    x[0] = 0
    x[1] = r
    x[2] = np.pi/2
    x[3] = 0

    v = np.empty_like(x)
    v[0] = du_a(a, r, m, q, ds2)
    v[1] = np.cos(a)
    v[2] = np.sin(a) / r
    v[3] = 0

norms = (g(x, m, q) * v[None, :, :] * v[:, None, :]).sum((0, 1))
assert np.allclose(norms, ds2)

start = default_timer()
x = np.array(list(islice(integrate(x, v, Γ, 1e-3, m, q), 10**4 * 2)))
print('{} seconds'.format(default_timer() - start))
print('{} geodesics'.format(x.shape[2]))
print('{} timesteps'.format(x.shape[0]))

r = x[:, 1, :]
θ = x[:, 2, :]

if True:
    mask = r < inner_horizon
    i = np.where(mask.any(0), mask.argmax(0), mask.shape[0])
    n = np.arange(mask.shape[0])
    r[n[:, None] > i[None, :]] = np.nan




axes = plt.subplot(111, projection='polar')
axes.set_facecolor('white')

a = np.linspace(0, 2 * np.pi, 100)
axes.plot(a, np.full(len(a), inner_horizon), linewidth=.5, color='grey', label='inner horizon')
axes.plot(a, np.full(len(a), outer_horizon), linewidth=.5, color='grey', label='outer horizon')
axes.plot(a, np.full(len(a), photon_sphere), linewidth=.5, color='grey', label='photon sphere')

axes.plot(θ, r, linewidth=.5, color='red')

axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
axes.set_ylim((0, 3))
axes.set_title('Reissner–Nördstrom metric in Eddington–Finkelstein coordinates')
#axes.legend()
axes.figure.tight_layout()
plt.show()
