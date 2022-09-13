from jax import grad, jit
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg, bicgstab

def hvp(f, x, y, v, a1, a2):
    return grad(lambda x, y: jnp.vdot(grad(f, argnums=a1)(x, y), v), argnums=a2)(x, y)

def CGD(f, g, x0, y0, n=0.2, iterations=50):
    x = [x0]
    y = [y0]
    for i in range(iterations):
        dfdx = grad(f, argnums=0)
        dgdy = grad(g, argnums=1)
        lx = x[-1]
        ly = y[-1]

        rhs_x = dfdx(lx, ly) - n * hvp(f, lx, ly, dgdy(lx, ly), 1, 0)
        def lhs_x(v):
            return v - n ** 2 * hvp(f, lx, ly, hvp(g, lx, ly, v, 0, 1), 1, 0)
        sol_x = cg(lhs_x, rhs_x)[0]

        rhs_y = dgdy(lx, ly) - n * hvp(g, lx, ly, dfdx(lx, ly), 0, 1)
        def lhs_y(v):
            return v - n ** 2 * hvp(g, lx, ly, hvp(f, lx, ly, v, 1, 0), 0, 1)
        sol_y = cg(lhs_y, rhs_y)[0]

        nx = lx - n * sol_x
        ny = ly - n * sol_y
        x.append(nx)
        y.append(ny)
    return x, y

def GDA(f, g, x0, y0, n=0.2, iterations=50):
    x = [x0]
    y = [y0]
    for i in range(iterations):
        dfdx = grad(f, argnums=0)
        dgdy = grad(g, argnums=1)
        nx = x[-1] - n * dfdx(x[-1], y[-1])
        ny = y[-1] - n * dgdy(x[-1], y[-1])
        x.append(nx)
        y.append(ny)
    return x, y