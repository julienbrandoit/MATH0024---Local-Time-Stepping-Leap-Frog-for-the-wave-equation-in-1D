"""
Author: BRANDOIT Julien

This code implements a finite element method (FEM) solution for a 1D Helmholtz
equation with complex solutions.

The code includes functions for generating exact solutions, assembling
stiffness matrices, and visualizing results.

Function documentation has been provided with the assistance of ChatGPT
to ensure clarity and consistency.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import fixed_quad
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, lil_matrix
from concurrent.futures import ProcessPoolExecutor

# == plotting functions ==

# Set the style of the plots
sns.set(style="whitegrid")
sns.set_context("paper")
sns_color = "magma"
resolution_plot = 2000


def plot_solution(x, u,
                  title=None, save_path=None, x_label=None,
                  y_label=None, marker='o', figsize=(16, 8), show=True,
                  legend=None, x_lim=None, y_lim=None, nodes=None,
                  use_loglog=False):
    """
    Plots the solution(s) over the given domain x.

    Parameters:
    - x : array-like, domain over which the solution is plotted.
    - u : array-like or list of arrays, solutions to be plotted.
    - title : str, optional, title of the plot.
    - save_path : str, optional, file path to save the plot as a PDF.
    - x_label, y_label : str, optional, axis labels.
    - marker : str, marker style for nodes.
    - figsize : tuple, figure size.
    - show : bool, whether to display the plot.
    - legend : list of str, optional, labels for each plotted solution.
    - x_lim, y_lim : tuple, optional, limits for x and y axes.
    - nodes : tuple, optional, nodes to highlight with markers.
    """

    _, ax = plt.subplots(figsize=figsize)

    if type(u) is not list:
        u = [u]

    colors = sns.color_palette(sns_color, len(u))

    for i in range(len(u)):
        if use_loglog:
            ax.plot(x, np.abs(u[i]), color=colors[i],
                    linewidth=3, label=legend[i] if legend else None)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
        else:
            ax.plot(x, u[i], color=colors[i],
                    linewidth=3, label=legend[i] if legend else None)

        # Plot node markers if nodes are provided
        if nodes is not None:
            if use_loglog:
                ax.plot(nodes[0][i], nodes[1][i], marker,
                        color=colors[i], markersize=8, label="_nolegend_")
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=2)
            else:
                ax.plot(nodes[0], np.real(nodes[1]) if i == 0
                        else np.imag(nodes[1]), marker,
                        color=colors[i], markersize=8, label="_nolegend_")

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if title is not None:
        ax.set_title(title, fontsize=20)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=20)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=20)
    if legend is not None:
        ax.legend(legend, loc='best', fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if show:
        plt.show()


def plot_solution_wave_space_time(x, t,
                                  k=1, omega=1, A=1, B=0,
                                  save_path=None):
    """
    Plots the solution of the wave equation in space and time.

    Parameters:
    - x : array-like, spatial domain.
    - t : array-like, temporal domain.
    - k : float, wavenumber.
    - omega : float, angular frequency.
    - A, B : float, coefficients for the solution.
    - save_path : str, optional, file path to save the plot as a PDF.
    """

    X, T = np.meshgrid(x, t)
    Z_real = np.real(solution_wave_space_time(X, T, k, omega, A, B))
    Z_imag = np.imag(solution_wave_space_time(X, T, k, omega, A, B))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    contour_real = ax[0].contourf(X, T, Z_real, cmap=sns_color)
    ax[0].set_title(r"Real part : $\Re\left\{e(x,t)\right\}$",
                    fontsize=20)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("t", fontsize=20)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 10)
    ax[0].xaxis.set_tick_params(labelsize=15)
    ax[0].yaxis.set_tick_params(labelsize=15)

    contour_imag = ax[1].contourf(X, T, Z_imag, cmap=sns_color)
    ax[1].set_title(r"Imaginary part : $\Im\left\{e(x,t)\right\}$",
                    fontsize=20)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("t", fontsize=20)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 10)
    ax[1].xaxis.set_tick_params(labelsize=15)
    ax[1].yaxis.set_tick_params(labelsize=15)

    cbar = fig.colorbar(contour_real, ax=ax, orientation='vertical')
    cbar.set_label("Amplitude", fontsize=20)
    cbar.ax.tick_params(labelsize=15)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()

# == exact_solution ==


def solution_wave_space_time(x, t, k=1, omega=1, A=1, B=0):
    """
    Computes the solution of the wave equation in space and time.

    Parameters:
    - x : array-like, spatial domain.
    - t : array-like, temporal domain.
    - k : float, wavenumber.
    - omega : float, angular frequency.
    - A, B : float, coefficients for the solution.

    Returns:
    - np.ndarray of complex values representing the solution over x and t.
    """

    return A * np.cos(-omega * t) * np.cos(k * x) +\
        1j * B * np.sin(-omega * t) * np.sin(k * x) + \
        1j * A * np.sin(-omega * t) * np.sin(k * x) + \
        B * np.cos(-omega * t) * np.cos(k * x)


def exact_solution(x, k):
    """
    Computes the exact solution for a 1D Helmholtz equation.

    Parameters:
    - x : array-like, domain over which to compute the solution.
    - k : float, wavenumber for the Helmholtz equation.

    Returns:
    - np.ndarray of complex values representing the exact solution over x.
    """

    return np.cos(k * x) + 1j * np.sin(k * x)


def derivative_exact_solution(x, k):
    """
    Computes the derivative of the exact solution for a 1D Helmholtz equation.

    Parameters:
    - x : array-like, domain over which to compute the solution.
    - k : float, wavenumber for the Helmholtz equation.

    Returns:
    - np.ndarray of complex values representing the derivative of the exact
    solution over x.
    """

    return -k * np.sin(k * x) + 1j * k * np.cos(k * x)

# == FEM implementation ==


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
    - k : float, wavenumber of the Helmholtz equation.
    """

    def __init__(self, nodes, h, k):
        # the local idx of the nodes is the idx of the node in the list
        self.nodes = nodes
        self.h = h
        self.k = k

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

        v = np.zeros((2, *x.shape), dtype=complex)

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

    def stiffness_matrix(self):
        """
        Constructs the local stiffness matrix for the element.
        # $$K_{ij} = ik\\left.[\varphi_j\varphi_i]\right|_{x=1}
        # + \\int_0^1\\dfrac{d \varphi_j}{dx}(x)\\dfrac{d \varphi_i}{dx}(x)dx
        # + k^2 \\int_0^1 \varphi_j(x) \varphi_i(x) dx$$

        Returns:
        - np.ndarray, 2x2 complex stiffness matrix.
        """

        K = np.zeros((2, 2), dtype=complex)

        phi_1 = self.phi(1.)
        K[0, 0] = 1j * self.k * phi_1[0] * phi_1[0]
        K[0, 1] = 1j * self.k * phi_1[0] * phi_1[1]
        K[1, 0] = 1j * self.k * phi_1[1] * phi_1[0]
        K[1, 1] = 1j * self.k * phi_1[1] * phi_1[1]

        K[0, 0] += -self.int_dphi_dphi(0, 0)
        K[0, 1] += -self.int_dphi_dphi(0, 1)
        K[1, 0] += -self.int_dphi_dphi(1, 0)
        K[1, 1] += -self.int_dphi_dphi(1, 1)

        K[0, 0] += self.k**2 * self.int_phi_phi(0, 0)
        K[0, 1] += self.k**2 * self.int_phi_phi(0, 1)
        K[1, 0] += self.k**2 * self.int_phi_phi(1, 0)
        K[1, 1] += self.k**2 * self.int_phi_phi(1, 1)

        return K


class FEM1D:
    def __init__(self, n_elements, k, u_0=1., L=1.):
        """
        Solves the 1D Helmholtz equation using the FEM.
        The domain is assumed to be between 0 and L : 0 <= x <= L.

        Attributes:
        - n_elements : positive int, the number of elements in the mesh.
        - k : float, wavenumber for the Helmholtz equation.
        - u_0 : complex, Dirichlet boundary condition at the first node.
        - L : float, length of the domain.
        """

        if n_elements <= 0:
            raise ValueError("n_elements (the number of elements) \
                             should be greater than 0")

        n_nodes = n_elements + 1
        self.nodes = [Node(x, i)
                      for i, x in enumerate(np.linspace(0, L, n_nodes))]
        self.elements = [Element1D([self.nodes[i],
                                    self.nodes[i + 1]],
                                   L / n_elements,
                                   k) for i in range(n_elements)]
        self.n_elements = n_elements
        self.n_nodes = n_nodes
        self.u_0 = u_0

    def assemble(self):
        """
        Assembles the global stiffness matrix by
        summing local element matrices.

        Returns:
        - csr_matrix, assembled stiffness matrix.
        """

        K_i = []
        K_j = []
        K_v = []

        for element in self.elements:
            local_stiffness = element.stiffness_matrix()
            node_ids = [element.nodes[0].global_idx,
                        element.nodes[1].global_idx]

            K_i.extend([node_ids[0], node_ids[0],
                        node_ids[1], node_ids[1]])
            K_j.extend([node_ids[0], node_ids[1],
                        node_ids[0], node_ids[1]])
            K_v.extend([local_stiffness[0, 0], local_stiffness[0, 1],
                        local_stiffness[1, 0], local_stiffness[1, 1]])

        return csr_matrix((K_v, (K_i, K_j)),
                          shape=(self.n_nodes, self.n_nodes))

    def solve(self):
        """
        Solves the FEM system, applying the Dirichlet boundary condition.

        Returns:
        - np.ndarray, solution vector at nodes.
        """

        K = self.assemble()

        u = np.zeros(self.n_nodes, dtype=complex)
        # Apply the Dirichlet boundary condition at the first node
        u[0] = self.u_0

        # Create the right-hand side vector (RHS)
        # RHS is -K times the boundary condition,
        # this is equivalent to [K_not_0] @ u = 0
        b = -K[:, 0] * self.u_0

        # Reduced stiffness matrix (excluding the first node)
        K_reduced = K[1:, 1:]

        # Reduced RHS (excluding the contribution from the first node)
        b_reduced = b[1:]

        u[1:] = spsolve(K_reduced, b_reduced)

        return u

    def sol(self, x):
        """
        Evaluates the FEM solution at arbitrary points using
        the Galerkin approximation : $u(x) = \\sum_{i=1}^{N} \\phi_i(x) u_i$.

        Parameters:
        - x : array-like, points at which to evaluate the solution.

        Returns:
        - tuple (solution values, (node coordinates, solution at nodes))
        """

        x = np.atleast_1d(x)
        u = self.solve()  # nodes values

        u_vals = np.zeros(x.shape, dtype=complex)

        for element in self.elements:
            phi = element.phi(x)

            for local_idx, node in enumerate(element.nodes):
                u_vals += phi[local_idx] * u[node.global_idx]

        if x.size > 1:
            return u_vals, ([n.x for n in self.nodes], u)
        else:
            return u_vals[0], ([n.x for n in self.nodes], u)

# == Part c) Convergence Analysis ==


def errorL2(args):
    """
    Computes the L2 error between the FEM solution and the exact solution.

    Parameters:
    - args : tuple, arguments for the error computation (mu, k, x0, x1, res).

    Returns:
    - float, L2 error between the FEM and exact solutions
    """

    mu, k, x0, x1, res = args
    fem = FEM1D(mu, k)
    u_fem, _ = fem.sol(np.linspace(x0, x1, res))
    u_exact = exact_solution(np.linspace(x0, x1, res), k)
    return np.sqrt(np.trapz(np.abs(u_fem - u_exact)**2, dx=(x1 - x0) / res))


def convergence_analysis(k, mu_h, x0=0, x1=1, res=2000):
    """
    Conducts a convergence analysis for the FEM solution.

    Parameters:
    - k : float, wavenumber for the Helmholtz equation.
    - mu_h : array-like, number of elements to test for convergence.
    - x0, x1 : float, domain limits.
    - res : int, resolution for the solution.

    Returns:
    - np.ndarray, errors for each number of elements in mu_h.
    """

    errors = []
    # I do multi processing to speed up the computation
    with ProcessPoolExecutor() as executor:
        for error_val in executor.map(errorL2, [(mu, k, x0, x1, res)
                                                for mu in mu_h]):
            errors.append(error_val)

    return np.array(errors)


if __name__ == "__main__":
    x = np.linspace(0, 1, resolution_plot)

    # == Part a) Exact Solution ==
    u_exact_k_pi = exact_solution(x, np.pi)
    plot_solution(x, [np.real(u_exact_k_pi), np.imag(u_exact_k_pi)],
                  title=r"$k=\pi$", x_label=r"$x$",
                  y_label=r"$u(x) = \Re\{u(x)\} + i \Im\{u(x)\}$",
                  show=True, legend=[r"$\Re\{u(x)\}$", r"$\Im\{u(x)\}$"],
                  x_lim=[0, 1], save_path="fig1_a.pdf")

    u_exact_k_7pi = exact_solution(x, 7 * np.pi)
    plot_solution(x, [np.real(u_exact_k_7pi), np.imag(u_exact_k_7pi)],
                  title=r"$k=7\pi$", x_label=r"$x$",
                  y_label=r"$u(x) = \Re\{u(x)\} + i \Im\{u(x)\}$",
                  show=True, legend=[r"$\Re\{u(x)\}$", r"$\Im\{u(x)\}$"],
                  x_lim=[0, 1], save_path="fig1_b.pdf")

    # == Part a) Wave space-time plot ==
    x = np.linspace(0, 1, resolution_plot)
    t = np.linspace(0, 10, resolution_plot)

    plot_solution_wave_space_time(x, t, k=1, omega=1, A=1, B=1j,
                                  save_path="wave_space_time_B_i.pdf")
    plot_solution_wave_space_time(x, t, k=1, omega=1, A=1, B=1,
                                  save_path="wave_space_time_B_1.pdf")

    # == Part b) FEM Approximation ==
    num_elements = 10

    k = np.pi
    fem = FEM1D(num_elements, k)
    u_fem_k_pi, u_nodes_k_pi = fem.sol(x)
    plot_solution(x, [np.real(u_fem_k_pi), np.imag(u_fem_k_pi)],
                  title=r"$k=\pi$", x_label=r"$x$",
                  y_label=r"$u_{FEM}(x) = \Re\{u_{FEM}(x)\} \
                            + i \Im\{u_{FEM}(x)\}$",
                  show=True,
                  legend=[r"$\Re\{u_{FEM}(x)\}$", r"$\Im\{u_{FEM}(x)\}$"],
                  x_lim=[0, 1], save_path="fig2_a.pdf", nodes=u_nodes_k_pi)

    k = 7 * np.pi
    fem = FEM1D(num_elements, k)
    u_fem_k_7pi, u_nodes_k_7pi = fem.sol(x)
    plot_solution(x, [np.real(u_fem_k_7pi), np.imag(u_fem_k_7pi)],
                  title=r"$k=7\pi$", x_label=r"$x$",
                  y_label=r"$u_{FEM}(x) = \Re\{u_{FEM}(x)\} \
                    + i \Im\{u_{FEM}(x)\}$",
                  show=True,
                  legend=[r"$\Re\{u_{FEM}(x)\}$", r"$\Im\{u_{FEM}(x)\}$"],
                  x_lim=[0, 1], save_path="fig2_b.pdf", nodes=u_nodes_k_7pi)

    # == Part c) Convergence Analysis ==

    mu_h = np.array([2**i for i in range(1, 20)])

    k = np.pi
    error_k_pi = convergence_analysis(k, mu_h)

    k = 7 * np.pi
    error_k_7pi = convergence_analysis(k, mu_h)

    plot_solution(mu_h, [error_k_pi, error_k_7pi],
                  x_label=r"Number of elements $\mu_h$",
                  y_label=r"L2 Error between $u_{\text{exact}}$ \
                    and $u_{\text{FEM}}$",
                  show=True, legend=[r"$k=\pi$", r"$k=7\pi$"],
                  save_path="fig3.pdf",
                  use_loglog=True, marker='o',
                  nodes=[[mu_h, mu_h], [error_k_pi, error_k_7pi]],
                  x_lim=[mu_h[0], mu_h[-1]])
