"""
Author: BRANDOIT Julien

TODO

Function documentation has been provided with the assistance of ChatGPT
to ensure clarity and consistency.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import fixed_quad
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, diags
from concurrent.futures import ProcessPoolExecutor
from matplotlib import animation

# == plotting functions ==

# Set the style of the plots
sns.set(style="whitegrid")
sns.set_context("paper")
sns_color = "magma"
resolution_plot = 2000

# == FEM1D implementation ==

class Node:
    """
    Represents a 1D node in the FEM mesh with spatial and index attributes.

    Attributes:
    - x : float, position of the node.
    - global_idx : int, global index of the node in the mesh.
    """

    def __init__(self, x, global_idx):
        self.x = x
        self.global_idx = global_idx

class Element1D:
    """
    Represents a 1D finite element for FEM computations.

    Attributes:
    - nodes : list, the two nodes defining the element.
    - h : float, the element length.
    """

    def __init__(self, nodes, h):
        # the local idx of the nodes is the idx of the node in the list
        self.nodes = nodes
        self.h = h

    def phi(self, x):
        """
        Piece-wise linear polynomial shape functions for the element
        evaluated at x.

        Parameters:
        - x : float or array-like, points where shape functions are evaluated.

        Returns:
        - np.ndarray, shape function values.
        """

        x = np.atleast_1d(x)

        # Create a mask to exclude points outside the element
        mask = (x >= self.nodes[0].x) & (x <= self.nodes[1].x)

        v = np.zeros((2, *x.shape))

        v[0, mask] = 1 / self.h * (self.nodes[1].x - x[mask])
        v[1, mask] = 1 / self.h * (x[mask] - self.nodes[0].x)

        return v if x.size > 1 else v[:, 0]

    def int_phi_phi(self, i, j):
        """
        Computes the integral of product of two shape functions using
        Gaussian quadrature.

        Parameters:
        - i, j : int, indices of the shape functions.

        Returns:
        - float, integral result.
        """

        def integrand(x):
            phi = self.phi(x)
            return phi[i] * phi[j]

        return fixed_quad(integrand, self.nodes[0].x, self.nodes[1].x, n=2)[0]

    def int_dphi_dphi(self, i, j):
        """
        Computes the integral of the derivatives of two shape functions.
        Since we have linear shape functions, the derivative is a constant.
        If i == j, the derivative is 1/h, otherwise it is -1/h.

        Parameters:
        - i, j : int, indices of the shape functions.

        Returns:
        - float, integral result.
        """

        if i == j:
            return 1 / self.h
        else:
            return -1 / self.h

    def mass_stiffness_matrices(self):
        """
        TODO
        """

        K = np.zeros((2, 2))
        M_lumped = np.zeros((2,)) # Lumping the mass matrix : M_ij = delta_ij * sum_k M_ik

        for i in range(2):
            for j in range(2):
                K[i, j] = self.int_dphi_dphi(i, j)
                M_lumped[i] += self.int_phi_phi(i, j)

        return K, M_lumped

class FEM1D:
    """
    TODO
    """

    def __init__(self, n_elements, c, L=1.):
        """
        TODO
        """
        if n_elements <= 0:
            raise ValueError("n_elements (the number of elements) \
                             should be greater than 0")

        n_nodes = n_elements + 1
        self.nodes = [Node(x, i)
                      for i, x in enumerate(np.linspace(0, L, n_nodes))]
        self.elements = [Element1D([self.nodes[i],
                                    self.nodes[i + 1]],
                                   L / n_elements
                                   ) for i in range(n_elements)]
        
        self.n_elements = n_elements
        self.n_nodes = n_nodes

        self.c = c
        self.L = L

    def assemble(self):
        """
        Assembles the stiffness matrix and the mass matrix for the FEM system.

        Returns:
        - np.ndarray, the assembled stiffness matrix.
        - np.ndarray, the assembled mass matrix.
        """

        K_i = []
        K_j = []
        K_v = []

        M_v = np.zeros(self.n_nodes)

        for element in self.elements:
            K, M_lumped = element.mass_stiffness_matrices()

            for i in range(2):
                M_v[element.nodes[i].global_idx] += M_lumped[i]

                for j in range(2):
                    K_i.append(element.nodes[i].global_idx)
                    K_j.append(element.nodes[j].global_idx)
                    K_v.append(self.c**2 * K[i, j])

        # K is a sparse matrix
        # M is a diagonal matrix, encoded using diags for efficient matrix-vector multiplication
        K = csr_matrix((K_v, (K_i, K_j)), shape=(self.n_nodes, self.n_nodes))
        M = diags(M_v)

        return K, M
    
class ClassicalFEM1DLeapFrog():
    """
    TODO
    """

    def __init__(self, n_elements, dt, c, L=1., u0 = lambda x: 0., v0 = lambda x: 0.):
        """
        Initializes the FEM1D leapfrog solver.
        
        Parameters:
        - n_elements : int, number of elements in the FEM mesh.
        - dt : float, time step size.
        - c : float, wave speed.
        - L : float, length of the domain.
        - u0 : callable, initial displacement function.
        - v0 : callable, initial velocity function.
        """

        self.fem = FEM1D(n_elements, c, L)
        self.dt = dt
        self.c = c
        self.L = L

        self.u = np.array([u0(node.x) for node in self.fem.nodes])
        v = np.array([v0(node.x) for node in self.fem.nodes])
        self.u_previous = np.array([u0(node.x + v[i] * dt) for i, node in enumerate(self.fem.nodes)])

        K, M = self.fem.assemble()
        M = M.sqrt()
        self.M_bar_sqrt_inv = M.copy().power(-1)
        self.A = self.M_bar_sqrt_inv @ K @ self.M_bar_sqrt_inv

        self.z = M @ self.u
        self.z_previous = M @ self.u_previous

    def step(self, update_u = True):
        """
        TODO
        """

        temp = self.z.copy()
        self.z = -self.z_previous + (- self.dt**2 * self.A) @ self.z + 2 * self.z
        self.z_previous = temp

        if update_u:
            self.u = self.M_bar_sqrt_inv @ self.z
            self.u_previous = self.M_bar_sqrt_inv @ self.z_previous

    def solve(self, T):
        """
        TODO
        """

        solution = []

        n_steps = int(T / self.dt)
        for _ in range(n_steps):
            self.step()
            solution.append(self.u.copy())

        return solution
    
class LocalTimeSteppingFEM1DLeapFrog():
    """
    TODO
    """
    pass

if __name__ == "__main__":

    c = -1.0
    L = 4.0
    sigma = 0.4
    number_of_elements = 2**5
    dt = 1/number_of_elements

    def u0(x, sigma=1., L=1.):
        return 1./(np.sqrt(2*np.pi)*sigma) * np.exp(-(x - L/2)**2/(2*sigma**2))

    cf1dp = ClassicalFEM1DLeapFrog(n_elements=number_of_elements, dt=dt, c=c, L=L, u0=lambda x : u0(x, sigma, L), v0=lambda x: c)

    temp = cf1dp.u_previous.copy()

    # plot u and u_previous
    fig, ax = plt.subplots()
    ax.plot([node.x for node in cf1dp.fem.nodes], cf1dp.u, label='u')
    ax.plot([node.x for node in cf1dp.fem.nodes], cf1dp.u_previous, label='u_previous')
    ax.legend()
    plt.show()

    # do one step
    cf1dp.step()

    # plot u and u_previous and u_previous_previous (temp)

    fig, ax = plt.subplots()
    ax.plot([node.x for node in cf1dp.fem.nodes], cf1dp.u, label='u')
    ax.plot([node.x for node in cf1dp.fem.nodes], cf1dp.u_previous, label='u_previous')
    ax.plot([node.x for node in cf1dp.fem.nodes], temp, label='u_previous_previous')
    ax.legend()
    plt.show()

    # solve the problem for 32 seconds and make an animation of the solution
    T = 10.
    solution = cf1dp.solve(T)

    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(-0.5, 1.5)
    line, = ax.plot([], [], lw=2)
    ax.set_title('Wave equation solution')
    ax.set_xlabel('x')
    ax.set_ylabel('u')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data([node.x for node in cf1dp.fem.nodes], solution[i])
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(solution), interval=dt*1000, blit=True)
    
    plt.show()
    plt.close()
    # Save the animation as a GIF
    anim.save('wave_solution.gif', writer='imagemagick', fps=int(1/dt))
