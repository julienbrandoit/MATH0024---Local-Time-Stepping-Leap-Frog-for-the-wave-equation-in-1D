"""
Author: BRANDOIT Julien

This file contains the implementation of the Finite Element Method (FEM) for solving
1D wave propagation problems. The FEM is implemented using the classical Leap-Frog
scheme and the local time-stepping Leap-Frog scheme.

Function documentation has been provided with the assistance of ChatGPT
to ensure clarity and consistency. I really insist on the fact that the core
of the code has been written by me. Only the function with clear references
to AI assistance have been generated with the help of ChatGPT or GitHub Copilot.
"""

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import fixed_quad
from scipy.sparse import csr_matrix, diags
from matplotlib.patches import FancyArrow
from tqdm import tqdm

# Set the style of the plots
sns.set(style="whitegrid")
sns.set_context("paper")
sns_color = "Spectral"
sns.set_palette(sns_color)

latex_preamble = r"\usepackage{amsmath}"
mpl.rcParams.update({
    # Use TeX for rendering text
    "text.usetex": True,
    "font.family": "serif",
    # Replace with your preferred serif font if needed
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": latex_preamble,

    # Axes settings
    "axes.titlesize": 14,     # Font size for axes titles
    "axes.labelsize": 12,     # Font size for axes labels

    # Tick settings
    "xtick.labelsize": 10,    # Font size for x-axis tick labels
    "ytick.labelsize": 10,    # Font size for y-axis tick labels

    # Legend settings
    "legend.fontsize": 10,    # Font size for legend

    # Figure settings
    "figure.figsize": (16, 8),  # Default figure size (width, height) in inches
    "figure.dpi": 100,        # Dots per inch

    # Lines
    "lines.linewidth": 2,     # Line width
    "lines.markersize": 6,    # Marker size

    # Grid settings
    "grid.color": "gray",     # Grid color
    "grid.linestyle": "--",   # Grid line style
    "grid.linewidth": 0.5,    # Grid line width
    "grid.alpha": 0.7         # Grid transparency
})


# == Utility functions ==

def scale_in_place_csr_matrix(K, M_bar_sqrt_inv):
    """
    Do the A = M_bar_sqrt_inv @ K @ M_bar_sqrt_inv in place. Leveraging the CSR format and the fact that
    the matrix M_bar_sqrt_inv is diagonal.

    Parameters:
    - K : csr_matrix, the stiffness matrix.
    - M_bar_sqrt_inv : csr_matrix, must be a diagonal matrix.

    Returns:
    - csr_matrix, the modified stiffness matrix K.
    """

    """
    Since M_bar_sqrt_inv is a diagonal matrix, doing M_bar_sqrt_inv @ K @ M_bar_sqrt_inv is equivalent to scaling each non-zero element of K
    by M_bar_sqrt_inv[i, i] * M_bar_sqrt_inv[j, j] where i and j are the row and column indices of the non-zero element in K.

    In order to be really memory efficient, we will modify the data attribute of the csr_matrix K in place instead of creating a new matrix
    and then discarding the K.
    """

    diag_M = M_bar_sqrt_inv.data.squeeze()

    # Iterate over all the non-zero elements of K using vectorized operations
    row_indices, col_indices = K.nonzero()
    K.data *= diag_M[row_indices] * diag_M[col_indices]

    return K  # K is modified in place

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

    def dphi(self, x):
        """
        Derivative of the piece-wise linear polynomial shape functions
        for the element evaluated at x.

        Parameters:
        - x : float or array-like, points where the derivatives of shape functions are evaluated.

        Returns:
        - np.ndarray, derivative values of the shape functions.
        """

        x = np.atleast_1d(x)

        # Create a mask to exclude points outside the element
        mask = (x >= self.nodes[0].x) & (x <= self.nodes[1].x)

        v = np.zeros((2, *x.shape))

        v[0, mask] = 1 / self.h
        v[1, mask] = -1 / self.h

        return v if x.size > 1 else v[:, 0]

    def int_phi_phi(self, i, j):
        """
        Computes the integral of the product of two shape functions using
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
        Computes the integral of the product of the derivatives of two shape functions
        using Gaussian quadrature.

        Parameters:
        - i, j : int, indices of the shape functions.

        Returns:
        - float, integral result of the derivatives of the shape functions.
        """

        def integrand(x):
            dphi = self.dphi(x)
            return dphi[i] * dphi[j]

        return fixed_quad(integrand, self.nodes[0].x, self.nodes[1].x, n=1)[0]

    def mass_stiffness_matrices(self):
        """
        Computes the stiffness matrix and lumped mass matrix for the element.

        Returns:
        - K : np.ndarray, 2x2 matrix representing the stiffness matrix.
        - M_lumped : np.ndarray, 1D array representing the lumped mass matrix.
        """

        K = np.zeros((2, 2))
        # Lumping the mass matrix : M_ij = delta_ij * sum_k M_ik
        M_lumped = np.zeros((2,))

        for i in range(2):
            for j in range(2):
                K[i, j] = self.int_dphi_dphi(i, j)
                M_lumped[i] += self.int_phi_phi(i, j)

        return K, M_lumped


class ManualFEM1D:
    """
    A class for solving a 1D finite element model using the finite element method (FEM).
    It handles the creation of the mesh, assembling of the global stiffness and lumped mass matrices,
    and solving the system for the given problem.

    Attributes:
    - nodes : list of Node, nodes in the mesh.
    - elements : list of Element1D, elements in the mesh.
    - n_elements : int, number of elements in the mesh.
    - n_nodes : int, number of nodes in the mesh.
    - c : float, wave speed constant.
    - L : float, length of the domain.
    """

    def __init__(self, mesh, c):
        """
        Initializes the ManualFEM1D object with the given mesh and wave speed.

        Parameters:
        - mesh : list of float, the mesh points.
        - c : float, the wave speed.
        """
        L = mesh[-1]
        n_elements = len(mesh) - 1
        n_nodes = n_elements + 1

        self.nodes = [Node(x, i) for i, x in enumerate(mesh)]
        self.elements = [Element1D([self.nodes[i], self.nodes[i + 1]],
                                   self.nodes[i + 1].x - self.nodes[i].x) for i in range(n_elements)]

        self.n_elements = n_elements
        self.n_nodes = n_nodes

        self.c = c
        self.L = L

    def assemble(self):
        """
        Assembles the stiffness matrix and the lumped mass matrix for the FEM system.

        Returns:
        - np.ndarray, the assembled stiffness matrix.
        - np.ndarray, the assembled mass matrix.
        """

        K_i = []
        K_j = []
        K_v = []

        M_v = np.zeros(self.n_nodes)

        for element in tqdm(self.elements, desc="Assembling matrices"):
            K, M_lumped = element.mass_stiffness_matrices()

            for i in range(2):
                M_v[element.nodes[i].global_idx] += M_lumped[i]

                for j in range(2):
                    K_i.append(element.nodes[i].global_idx)
                    K_j.append(element.nodes[j].global_idx)
                    K_v.append(self.c**2 * K[i, j])

        # K is a sparse matrix
        # M is a diagonal matrix, encoded using diags for efficient
        # matrix-vector multiplication
        K = csr_matrix((K_v, (K_i, K_j)), shape=(self.n_nodes, self.n_nodes))
        M = diags(M_v)

        return K, M

    def get_solution(self, u, x):
        """
        Interpolates the solution at the point x using the shape functions.

        Parameters:
        - u : np.ndarray, the solution vector.
        - x : np.ndarray, the points where the solution is interpolated.

        Returns:
        - np.ndarray, the interpolated solution.
        """

        u = np.atleast_1d(u)
        x = np.atleast_1d(x)

        u_interp = np.zeros_like(x)

        for element in self.elements:
            mask = (x >= element.nodes[0].x) & (x <= element.nodes[1].x)
            u_interp[mask] = np.dot(element.phi(x[mask]),
                                    u[[node.global_idx for node in element.nodes]])

        return u_interp


class FEM1D(ManualFEM1D):
    """
    A class for solving a 1D finite element model using the finite element method (FEM).
    This class can be used to create a REGULAR mesh with a specified number of elements.

    Inherits from the ManualFEM1D class and sets up the finite element method for
    1D problems, allowing for the creation of mesh and assembling of matrices.

    Attributes:
    - User should refer to the ManualFEM1D class for the attributes.
    """

    def __init__(self, n_elements, c, L=1.):
        """
        Initializes the FEM1D object with the specified number of elements, wave speed, and domain length.

        Parameters:
        - n_elements : int, number of elements in the mesh.
        - c : float, the wave speed constant.
        - L : float, optional, the length of the domain (default is 1.0).

        Raises:
        - ValueError : if the number of elements is less than or equal to 0.
        """
        if n_elements <= 0:
            raise ValueError(
                "n_elements (the number of elements) should be greater than 0")

        n_nodes = n_elements + 1
        mesh = np.linspace(0, L, n_nodes)
        super().__init__(mesh, c)


class ClassicalFEM1DLeapFrog():
    """
    Implements the classical Leap-Frog scheme for solving 1D wave propagation problems
    using the finite element method (FEM).

    Attributes:
    - fem : FEM1D or ManualFEM1D, finite element method instance.
    - dt : float, the time step size.
    - c : float, the wave speed constant.
    - L : float, the length of the domain.
    - u : np.ndarray, the current solution vector at the nodes.
    - u_previous : np.ndarray, the solution vector at the previous time step.
    - M_bar_sqrt_inv : np.ndarray, the inverse square root of the mass matrix.
    - A : np.ndarray, the matrix that combines the stiffness and mass matrices.
    - z : np.ndarray, modified solution vector for Leap-Frog integration.
    - z_previous : np.ndarray, previous time step modified solution vector.
    """

    def __init__(self, n_elements_or_mesh, dt, c, L=1.,
                 u0=lambda x: 0., v0=lambda x: 0.):
        """
        Initializes the Leap-Frog scheme for solving the 1D wave equation with FEM.

        Parameters:
        - n_elements_or_mesh : int or list, number of elements or mesh points defining the mesh.
        - dt : float, the time step size.
        - c : float, the wave speed constant.
        - L : float, optional, the length of the domain (default is 1.0).
        - u0 : function, optional, initial displacement as a function of position (default is 0).
        - v0 : function, optional, initial velocity as a function of position (default is 0).
        """
        if isinstance(n_elements_or_mesh, int):
            n_elements = n_elements_or_mesh
            self.fem = FEM1D(n_elements, c, L)
        else:
            self.fem = ManualFEM1D(n_elements_or_mesh, c)

        self.dt = dt
        self.c = c
        self.L = L

        self.u = np.array([u0(node.x) for node in self.fem.nodes])
        self.u_previous = np.array(
            [u0(node.x + dt * v0(node.x)) for node in self.fem.nodes])

        K, M = self.fem.assemble()
        M = M.sqrt()
        self.z = M @ self.u
        self.z_previous = M @ self.u_previous

        self.M_bar_sqrt_inv = M.power(-1)
        self.A = scale_in_place_csr_matrix(K, self.M_bar_sqrt_inv)

    def step(self, update_u=True):
        """
        Performs one time step using the Leap-Frog scheme.

        Parameters:
        - update_u : bool, optional, if True, the solution vector u is updated.

        Updates the solution vector `u` for the current time step using the Leap-Frog method.
        """
        temp = self.z.copy()
        self.z = -self.z_previous + \
            (- self.dt**2 * self.A) @ self.z + 2 * self.z
        self.z_previous = temp

        if update_u:
            self.u = self.M_bar_sqrt_inv @ self.z
            self.u_previous = self.M_bar_sqrt_inv @ self.z_previous

    def solve(self, T, t_eval=None):
        """
        Solves the 1D wave equation over a specified time interval using the Leap-Frog scheme.

        Parameters:
        - T : float, total time for the simulation.
        - t_eval : np.ndarray or list, optional, specific time steps at which to store the solution.

        Returns:
        - solution : list of np.ndarray, the solution at the requested time steps.
        """
        if t_eval is None:
            t_eval = np.arange(0, T + self.dt, self.dt)
        else:
            t_eval = np.atleast_1d(t_eval)

        solution = []

        n_steps = int(np.ceil(T / self.dt)) + 1

        if t_eval[0] == 0:
            solution.append(self.u.copy())
            t_eval = t_eval[1:]

        for i in tqdm(range(1, n_steps),
                      desc="Solving the classical Leap-Frog scheme"):
            if i == n_steps - \
                    1 or (len(t_eval) >= 2 and t_eval[0] <= i * self.dt < t_eval[1]):
                self.step(update_u=True)
                solution.append(self.u.copy())
                t_eval = t_eval[1:]
            else:
                self.step(update_u=False)

            if len(t_eval) == 0:
                break

        return solution

    def get_solution(self, u, x):
        """
        Interpolates the solution at specified positions.

        Parameters:
        - u : np.ndarray, the solution vector at the nodes.
        - x : np.ndarray, the positions at which to interpolate the solution.

        Returns:
        - np.ndarray, the interpolated solution at the specified positions.
        """
        return self.fem.get_solution(u, x)


class LocalTimeSteppingFEM1DLeapFrog():
    """
    Implements a local time-stepping Leap-Frog scheme for solving 1D wave propagation problems
    using the finite element method (FEM) with different time steps for coarse and fine nodes.

    Attributes:
    - fem : ManualFEM1D, finite element method instance.
    - dt : float, the global time step size.
    - p : int, the number of sub-steps for fine nodes.
    - u : np.ndarray, the current solution vector at the nodes.
    - u_previous : np.ndarray, the solution vector at the previous time step.
    - M_bar_sqrt_inv : np.ndarray, the inverse square root of the mass matrix.
    - A_cc, A_cf, A_fc, A_ff : np.ndarray, sub-matrices of the matrix A for different node types.
    - z_c, z_f, z_previous_c, z_previous_f : np.ndarray, solution vectors for coarse and fine nodes.
    - coarse_idx, fine_idx : list, indices of coarse and fine nodes.
    """

    def __init__(self, mesh, mask_is_fine_node, dt, p,
                 c, u0=lambda x: 0., v0=lambda x: 0.):
        """
        Initializes the Local Time Stepping Leap-Frog scheme for solving 1D wave equations.

        Parameters:
        - mesh : list or np.ndarray, the mesh points defining the domain.
        - mask_is_fine_node : np.ndarray, boolean mask indicating fine nodes.
        - dt : float, the global time step size.
        - p : int, the number of sub-steps for fine nodes.
        - c : float, the wave speed constant.
        - u0 : function, optional, initial displacement as a function of position (default is 0).
        - v0 : function, optional, initial velocity as a function of position (default is 0).
        """
        mesh = np.atleast_1d(mesh)
        self.fem = ManualFEM1D(mesh, c)

        self.dt = dt
        self.p = p
        self.c = c
        self.L = mesh[-1]

        self.u = np.array([u0(node.x) for node in self.fem.nodes])
        self.u_previous = np.array(
            [u0(node.x + dt * v0(node.x)) for node in self.fem.nodes])

        K, M = self.fem.assemble()
        M = M.sqrt()

        z = M @ self.u
        z_previous = M @ self.u_previous

        self.M_bar_sqrt_inv = M.power(-1)
        A = scale_in_place_csr_matrix(K, self.M_bar_sqrt_inv)

        self.coarse_idx = [
            node.global_idx for node in self.fem.nodes if not mask_is_fine_node[node.global_idx]]
        self.fine_idx = [
            node.global_idx for node in self.fem.nodes if mask_is_fine_node[node.global_idx]]

        self.A_cc = A[self.coarse_idx, :][:, self.coarse_idx]
        self.A_cf = A[self.coarse_idx, :][:, self.fine_idx]
        self.A_fc = A[self.fine_idx, :][:, self.coarse_idx]
        self.A_ff = A[self.fine_idx, :][:, self.fine_idx]

        self.z_c = z[self.coarse_idx]
        self.z_f = z[self.fine_idx]
        self.z_previous_c = z_previous[self.coarse_idx]
        self.z_previous_f = z_previous[self.fine_idx]

    def step(self, update_u=True):
        """
        Performs one step of the local time-stepping Leap-Frog scheme.

        Parameters:
        - update_u : bool, optional, if True, the solution vector u is updated.

        Updates the solution vectors `u` for both coarse and fine nodes using the local time-stepping scheme.
        """
        w_c = self.A_cc @ self.z_c
        w_f = self.A_fc @ self.z_c

        z_0p_c = self.z_c.copy()
        z_0p_f = self.z_f.copy()

        APz_0p_c = self.A_cf @ z_0p_f
        APz_0p_f = self.A_ff @ z_0p_f

        z_1p_c = z_0p_c - 1 / 2 * (self.dt / self.p)**2 * (w_c + APz_0p_c)
        z_1p_f = z_0p_f - 1 / 2 * (self.dt / self.p)**2 * (w_f + APz_0p_f)

        z_m1p_c = z_0p_c
        z_m1p_f = z_0p_f

        z_mp_c = z_1p_c
        z_mp_f = z_1p_f

        for _ in range(2, self.p + 1):
            temp_c = z_mp_c.copy()
            temp_f = z_mp_f.copy()

            APz_mp_c = self.A_cf @ z_mp_f
            APz_mp_f = self.A_ff @ z_mp_f

            z_mp_c = 2 * z_mp_c - z_m1p_c - \
                (self.dt / self.p)**2 * (w_c + APz_mp_c)
            z_mp_f = 2 * z_mp_f - z_m1p_f - \
                (self.dt / self.p)**2 * (w_f + APz_mp_f)

            z_m1p_c = temp_c
            z_m1p_f = temp_f

        temp_c = self.z_c.copy()
        temp_f = self.z_f.copy()

        self.z_c = -self.z_previous_c + 2 * z_mp_c
        self.z_f = -self.z_previous_f + 2 * z_mp_f

        self.z_previous_c = temp_c
        self.z_previous_f = temp_f

        if update_u:
            self.u[self.coarse_idx] = self.z_c
            self.u[self.fine_idx] = self.z_f
            self.u = self.M_bar_sqrt_inv @ self.u

            self.u_previous[self.coarse_idx] = self.z_previous_c
            self.u_previous[self.fine_idx] = self.z_previous_f
            self.u_previous = self.M_bar_sqrt_inv @ self.u_previous

    def solve(self, T, t_eval=None):
        """
        Solves the local time-stepping Leap-Frog scheme over a specified time interval.

        Parameters:
        - T : float, total time for the simulation.
        - t_eval : np.ndarray or list, optional, specific time steps at which to store the solution.

        Returns:
        - solution : list of np.ndarray, the solution at the requested time steps.
        """
        if t_eval is None:
            t_eval = np.arange(0, T + self.dt, self.dt)
        else:
            t_eval = np.atleast_1d(t_eval)

        solution = []

        n_steps = int(np.ceil(T / self.dt)) + 1

        if t_eval[0] == 0:
            solution.append(self.u.copy())
            t_eval = t_eval[1:]

        for i in tqdm(range(1, n_steps),
                      desc="Solving the local time stepping Leap-Frog scheme"):
            if i == n_steps - \
                    1 or (len(t_eval) >= 2 and t_eval[0] <= i * self.dt < t_eval[1]):
                self.step(update_u=True)
                solution.append(self.u.copy())
                t_eval = t_eval[1:]
            else:
                self.step(update_u=False)

            if len(t_eval) == 0:
                break

        return solution

    def get_solution(self, u, x):
        """
        Interpolates the solution at specified positions.

        Parameters:
        - u : np.ndarray, the solution vector at the nodes.
        - x : np.ndarray, the positions at which to interpolate the solution.

        Returns:
        - np.ndarray, the interpolated solution at the specified positions.
        """
        return self.fem.get_solution(u, x)


def compare_time_evolution(
        problemSolver_as_a_function_of_dt, T, t_eval, dt_list, title):
    """
    Compare the time evolution of a solution for multiple time step sizes (dt).

    This function generates a plot comparing the time evolution of the solution
    at different time step sizes. Each subplot corresponds to a different time
    point, and each column represents a different time step size.

    Parameters:
    - problemSolver_as_a_function_of_dt: Function to generate the problem solver
      given dt. The function should return an instance of a solver class with a `solve` method.
    - T: Total simulation time.
    - t_eval: List of time points at which the solution is evaluated.
    - dt_list: List of time step sizes to compare.
    - title: Title of the plot (and file to save).

    Note:
    - The code for this function was generated with the assistance of ChatGPT and GitHub Copilot.

    """
    fig, ax = plt.subplots(
        len(t_eval), len(dt_list), figsize=(
            16, 20), sharex=True, sharey=True)
    for i, dt in enumerate(dt_list):
        print(f"Computing solution for dt = {dt}...")
        problem = problemSolver_as_a_function_of_dt(dt)
        solution = problem.solve(T, t_eval)
        print("... done. Plotting (can take some time) ...")
        for j, u in enumerate(solution):
            ax[j, i].plot([node.x for node in problem.fem.nodes],
                          u, marker="o", markersize=5, linewidth=2)
            ax[j, i].set_xlim(0, problem.L)
            ax[j, i].set_ylim(-0.1, 1.5)
            ax[j, i].xaxis.set_tick_params(labelsize=15)
            ax[j, i].yaxis.set_tick_params(labelsize=15)
            ax[j, i].set_yticks([])
            if j == len(t_eval) - 1:
                ax[j, i].set_xlabel(r"$x$", fontsize=20)
            if j == 0:
                ax[j, i].set_title(r"$\Delta t = $" + f"{dt:.4f}", fontsize=20)
            for spine in ax[j, i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

    fig.subplots_adjust(right=0.9)

    arrow_start_y = 0.875
    arrow_end_y = 0.075
    arrow_x = 0.92

    arrow = FancyArrow(
        x=arrow_x,
        y=arrow_start_y,
        dx=0,
        dy=arrow_end_y -
        arrow_start_y,
        width=0.001,
        length_includes_head=True,
        head_width=0.004,
        head_length=0.015,
        color="black",
        transform=fig.transFigure,
        clip_on=False)
    fig.patches.append(arrow)

    for i, axis in enumerate(ax[:, -1]):
        pos = axis.get_position()
        tick_y = pos.y0 + (pos.y1 - pos.y0) / 2
        fig.text(
            arrow_x + 0.005,
            tick_y,
            f"{t_eval[i]:.1f}",
            va="center",
            ha="left",
            fontsize=15)

        start_line = ax[i, 0].get_position().x0
        end_line = arrow_x + 0.0025
        fig.add_artist(plt.Line2D(
            [start_line, end_line],
            [tick_y, tick_y],
            color="black",
            linewidth=1.5,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False
        ))

    fig.text(
        arrow_x +
        0.01,
        arrow_end_y +
        0.02,
        "Time",
        ha="left",
        fontsize=16)

    plt.savefig(title, bbox_inches='tight')
    plt.show()


def plot_time_evolution_for_single_dt(
        problemSolver_as_a_function_of_dt, T, t_eval, dt, title):
    """
    Plot the time evolution of a solution for a single time step size (dt).

    This function generates a plot of the solution at different time points for
    a given time step size, displaying the evolution of the solution over time.

    Parameters:
    - problemSolver_as_a_function_of_dt: Function to generate the problem solver
      given dt. The function should return an instance of a solver class with a `solve` method.
    - T: Total simulation time.
    - t_eval: List of time points at which the solution is evaluated.
    - dt: Time step size for the simulation.
    - title: Title of the plot (and file to save).

    Note:
    - The code for this function was generated with the assistance of ChatGPT and GitHub Copilot.

    """
    fig, ax = plt.subplots(
        len(t_eval), 1, figsize=(
            16 / 3, 20), sharex=True, sharey=True)

    print(f"Computing solution for dt = {dt}...")
    problem = problemSolver_as_a_function_of_dt(dt)
    solution = problem.solve(T, t_eval)
    print("... done. Plotting (can take some time) ...")

    for j, u in enumerate(solution):
        ax[j].plot([node.x for node in problem.fem.nodes],
                   u, marker="o", markersize=5, linewidth=2)
        ax[j].set_xlim(0, problem.L)
        ax[j].set_ylim(-0.1, 1.5)
        ax[j].xaxis.set_tick_params(labelsize=15)
        ax[j].yaxis.set_tick_params(labelsize=15)
        ax[j].set_yticks([])
        if j == len(t_eval) - 1:
            ax[j].set_xlabel(r"$x$", fontsize=20)
        if j == 0:
            ax[j].set_title(r"$\Delta t = $" + f"{dt:.4f}", fontsize=20)
        for spine in ax[j].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    fig.subplots_adjust(right=0.9)

    arrow_start_y = 0.875
    arrow_end_y = 0.075
    arrow_x = 0.92

    arrow = FancyArrow(
        x=arrow_x,
        y=arrow_start_y,
        dx=0,
        dy=arrow_end_y -
        arrow_start_y,
        width=0.001,
        length_includes_head=True,
        head_width=0.004,
        head_length=0.015,
        color="black",
        transform=fig.transFigure,
        clip_on=False)
    fig.patches.append(arrow)

    for i, axis in enumerate(ax):
        pos = axis.get_position()
        tick_y = pos.y0 + (pos.y1 - pos.y0) / 2
        fig.text(
            arrow_x + 0.005,
            tick_y,
            f"{t_eval[i]:.1f}",
            va="center",
            ha="left",
            fontsize=15)

        start_line = ax[i].get_position().x0
        end_line = arrow_x + 0.0025
        fig.add_artist(plt.Line2D(
            [start_line, end_line],
            [tick_y, tick_y],
            color="black",
            linewidth=1.5,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False
        ))

    fig.text(
        arrow_x +
        0.01,
        arrow_end_y +
        0.02,
        "Time",
        ha="left",
        fontsize=16)

    plt.savefig(title, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # PROBLEM DEFINITION
    c = -1.0
    L = 4.0
    sigma = 0.4

    # SIMULATION PARAMETERS
    T = 9.0
    dt_eval = 0.5
    t_eval = np.arange(0, T + dt_eval, dt_eval)

    # MESH PARAMETERS
    h_coarse = 0.1  # should be chosen such that L/h_coarse is an integer
    n_elements_coarse = int(L / h_coarse)
    p = 4  # Should be an integer
    n_elements_refined = 2  # We refine in [1, 1+h_coarse*n_elements_refined]
    h_fine = h_coarse / p

    # Initial conditions
    def u0(x, sigma=1., L=1.):  # Initial displacement
        return 1. / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-(x - L / 2)**2 / (2 * sigma**2))

    def v0(x):  # Initial velocity
        return c

    # == Part 1: Initial setup and mesh generation ==
    print("\n == Part 1 ==\n")

    refined_mesh = []

    for i, x in enumerate(np.arange(0, L + h_coarse, h_coarse)):
        refined_mesh.append(x)
        if 1 <= x < 1 + h_coarse * n_elements_refined:
            for j in range(1, p):
                refined_mesh.append(x + j * h_coarse / p)

    regular_mesh_problem = ClassicalFEM1DLeapFrog(
        n_elements_or_mesh=n_elements_coarse,
        dt=0.1,
        c=c,
        L=L,
        u0=lambda x: u0(x, sigma, L),
        v0=v0)
    refined_mesh_problem = ClassicalFEM1DLeapFrog(
        n_elements_or_mesh=refined_mesh,
        dt=0.1,
        c=c,
        L=L,
        u0=lambda x: u0(x, sigma, L),
        v0=v0)

    # Plot initial conditions and mesh
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    y_mesh = 0
    for ax_i, problem in zip(ax, [regular_mesh_problem, refined_mesh_problem]):
        ax_i.hlines(y_mesh, 0, L, colors='black', linestyles='solid')
        for node in problem.fem.nodes:
            ax_i.plot(node.x, y_mesh, 'k', marker='d', markersize=5,
                      label="Node position" if node.global_idx == 0 else None)
            ax_i.vlines(node.x,
                        y_mesh,
                        max(problem.u[node.global_idx],
                            problem.u_previous[node.global_idx]),
                        colors='black',
                        linestyles='dashed',
                        alpha=0.5,
                        linewidth=1.5)

        ax_i.set_xlim(0, L)
        ax_i.set_ylim(-0.1, 1.2)
        ax_i.xaxis.set_tick_params(labelsize=15)
        ax_i.yaxis.set_tick_params(labelsize=15)
        ax_i.set_xlabel(r"$x$", fontsize=20)

    ax[0].plot([node.x for node in regular_mesh_problem.fem.nodes],
               regular_mesh_problem.u,
               marker="o",
               markersize=5,
               linewidth=2,
               label=r"$\mathbf{u}_0$")
    ax[0].plot([node.x for node in regular_mesh_problem.fem.nodes],
               regular_mesh_problem.u_previous,
               marker="o",
               markersize=5,
               linewidth=2,
               label=r"$\mathbf{u}_{-1}$")

    ax[1].plot([node.x for node in refined_mesh_problem.fem.nodes],
               refined_mesh_problem.u,
               marker="o",
               markersize=5,
               linewidth=2,
               label=r"$\mathbf{u}_0$")
    ax[1].plot([node.x for node in refined_mesh_problem.fem.nodes],
               refined_mesh_problem.u_previous,
               marker="o",
               markersize=5,
               linewidth=2,
               label=r"$\mathbf{u}_{-1}$")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=20, ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("initial_approximation.pdf", bbox_inches='tight')
    plt.show()

    # Plot of both meshes (regular is red and refined is blue)
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot([n.x for n in regular_mesh_problem.fem.nodes],
            np.zeros(len(regular_mesh_problem.fem.nodes)) + 1,
            'ro',
            markersize=10,
            alpha=0.5,
            label="Regular mesh")
    ax.plot([n.x for n in refined_mesh_problem.fem.nodes],
            np.zeros(len(refined_mesh_problem.fem.nodes)),
            'bo',
            markersize=10,
            alpha=0.5,
            label="Refined mesh")

    ax.set_xlim(0, L)
    ax.set_ylim(-0.1, 1.1)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.set_yticks([])
    ax.set_xlabel(r"$x$", fontsize=20)

    ax.legend(fontsize=15, loc='upper center',
              bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig("mesh_pts1.pdf", bbox_inches='tight')
    plt.show()

    # == Part 2: Stability criterion on the coarse mesh ==
    print("\n == Part 2 ==\n")
    dt_list = [h_coarse * 0.95, h_coarse, 1.05 * h_coarse]

    def regular_problemSolver_as_a_function_of_dt(dt): return ClassicalFEM1DLeapFrog(
        n_elements_or_mesh=n_elements_coarse, dt=dt, c=c, L=L, u0=lambda x: u0(
            x, sigma, L), v0=v0)
    compare_time_evolution(
        regular_problemSolver_as_a_function_of_dt,
        T,
        t_eval,
        dt_list,
        title="regular_mesh_sol.pdf")

    # == Part 3: Stability criterion on the refined mesh ==
    print("\n == Part 3 ==\n")
    dt_list = [h_fine * 0.95, h_fine, 1.05 * h_fine]

    def refined_problemSolver_as_a_function_of_dt(dt): return ClassicalFEM1DLeapFrog(
        n_elements_or_mesh=refined_mesh, dt=dt, c=c, L=L, u0=lambda x: u0(x, sigma, L), v0=v0)
    compare_time_evolution(
        refined_problemSolver_as_a_function_of_dt,
        T,
        t_eval,
        dt_list,
        title="refined_mesh_sol.pdf")

    # == Part 4: Local time stepping ==
    print("\n == Part 4 ==\n")

    refined_mesh = []
    mask_is_fine_node = []

    flag = False
    for i, x in enumerate(np.arange(0, L + h_coarse, h_coarse)):
        refined_mesh.append(x)
        if 1 <= x < 1 + h_coarse * n_elements_refined:
            mask_is_fine_node.append(True)
            for j in range(1, p):
                refined_mesh.append(x + j * h_coarse / p)
                mask_is_fine_node.append(True)
            flag = True
        elif flag:
            mask_is_fine_node.append(True)
            flag = False
        else:
            mask_is_fine_node.append(False)

    refined_mesh = np.array(refined_mesh)
    mask_is_fine_node = np.array(mask_is_fine_node)

    # Plot mesh with color coding for fine nodes
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(
        refined_mesh[mask_is_fine_node],
        np.zeros(
            len(refined_mesh))[mask_is_fine_node],
        'ro',
        markersize=10,
        alpha=0.5,
        label=r"Fine nodes $\mathbf{z}_{\text{F}}$")
    ax.plot(refined_mesh[~mask_is_fine_node],
            np.zeros(len(refined_mesh))[~mask_is_fine_node],
            'bo',
            markersize=10,
            alpha=0.5,
            label=r"Coarse nodes $\mathbf{z}_{\text{C}}$")

    ax.set_xlim(0, L)
    ax.set_ylim(-0.1, 0.1)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_yticks([])

    ax.legend(fontsize=15, loc='upper center',
              bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig("mesh_pts2.pdf", bbox_inches='tight')
    plt.show()

    # Study local time stepping method on the refined mesh
    dt_list = [h_coarse * 0.95, h_coarse, 1.05 * h_coarse]

    def local_time_stepping_problemSolver_as_a_function_of_dt(dt): return LocalTimeSteppingFEM1DLeapFrog(
        mesh=refined_mesh, mask_is_fine_node=mask_is_fine_node, dt=dt, p=p, c=c, u0=lambda x: u0(x, sigma, L), v0=v0)
    compare_time_evolution(
        local_time_stepping_problemSolver_as_a_function_of_dt,
        T,
        t_eval,
        dt_list,
        title="local_time_stepping_sol_coarse.pdf")

    dt_list = [h_fine * 0.95, h_fine, 1.05 * h_fine]
    compare_time_evolution(
        local_time_stepping_problemSolver_as_a_function_of_dt,
        T,
        t_eval,
        dt_list,
        title="local_time_stepping_sol_fine.pdf")
