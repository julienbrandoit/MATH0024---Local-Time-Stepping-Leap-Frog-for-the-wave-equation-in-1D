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
from scipy.sparse import csr_matrix, diags
from matplotlib.patches import FancyArrow
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

# Set the style of the plots
sns.set(style="whitegrid")
sns.set_context("paper")
sns_color = "Spectral"
sns.set_palette(sns_color)
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

class ManualFEM1D:
    """
    TODO
    """

    def __init__(self, mesh, c):
        """
        TODO
        """
        L = mesh[-1]
        n_elements = len(mesh) - 1
        n_nodes = n_elements + 1

        self.nodes = [Node(x, i) for i, x in enumerate(mesh)]
        self.elements = [Element1D([self.nodes[i], self.nodes[i + 1]], self.nodes[i + 1].x - self.nodes[i].x) for i in range(n_elements)]

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

        for element in tqdm(self.elements, desc="Assembling matrices"):
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
            u_interp[mask] = np.dot(element.phi(x[mask]), u[[node.global_idx for node in element.nodes]])

        return u_interp

class FEM1D(ManualFEM1D):
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
        mesh = np.linspace(0, L, n_nodes)
        super().__init__(mesh, c)

# == Leap-Frog scheme implementation ==

class ClassicalFEM1DLeapFrog():
    """
    TODO
    """

    def __init__(self, n_elements_or_mesh, dt, c, L=1., u0 = lambda x: 0., v0 = lambda x: 0.):
        """
        TODO
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
        self.u_previous = np.array([u0(node.x + dt * v0(node.x)) for node in self.fem.nodes])

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

    def solve(self, T, t_eval=None):
        """
        TODO
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

        for i in tqdm(range(1, n_steps), desc="Solving the classical Leap-Frog scheme"):
            # check if we need to store the solution that is if we are
            # at a time in t_eval or if we are such that previous step < t_eval < current step
            if i == n_steps - 1 or (len(t_eval) >= 2 and t_eval[0] <= i * self.dt < t_eval[1]):
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
        TODO
        """

        return self.fem.get_solution(u, x)

class LocalTimeSteppingFEM1DLeapFrog():
    """
    TODO
    """

    def __init__(self, mesh, mask_is_fine_node, dt, p, c, u0 = lambda x: 0., v0 = lambda x: 0.):
        """
        TODO
        """

        mesh = np.atleast_1d(mesh)
        self.mask_is_fine_node = mask_is_fine_node
        self.fem = ManualFEM1D(mesh, c)
        
        self.dt = dt
        self.p = p
        self.c = c

        self.u = np.array([u0(node.x) for node in self.fem.nodes])
        self.u_previous = np.array([u0(node.x + dt * v0(node.x)) for node in self.fem.nodes])
        
        K, M = self.fem.assemble()
        M = M.sqrt()
        self.M_bar_sqrt_inv = M.copy().power(-1)
        A = self.M_bar_sqrt_inv @ K @ self.M_bar_sqrt_inv

        self.coarse_idx = [node.global_idx for node in self.fem.nodes if not self.mask_is_fine_node[node.global_idx]]
        self.fine_idx = [node.global_idx for node in self.fem.nodes if self.mask_is_fine_node[node.global_idx]]

        # A = [[A_cc, A_cf], [A_fc, A_ff]]
        # z = [z_c, z_f]

        self.A_cc = A[self.coarse_idx, :][:, self.coarse_idx]
        self.A_cf = A[self.coarse_idx, :][:, self.fine_idx]
        self.A_fc = A[self.fine_idx, :][:, self.coarse_idx]
        self.A_ff = A[self.fine_idx, :][:, self.fine_idx]

        z = M @ self.u
        z_previous = M @ self.u_previous

        self.z_c = z[self.coarse_idx]
        self.z_f = z[self.fine_idx]
        self.z_previous_c = z_previous[self.coarse_idx]
        self.z_previous_f = z_previous[self.fine_idx]

    def step(self, update_u = True):
        """
        TODO
        """

        # Application of A onto the coarse nodes
        w_c = self.A_cc @ self.z_c
        w_f = self.A_fc @ self.z_c

        # Update the fine nodes (p sub-steps)

        z_0p_c = self.z_c.copy()
        z_0p_f = self.z_f.copy()

        APz_0p_c = self.A_cf @ z_0p_f
        APz_0p_f = self.A_ff @ z_0p_f

        z_1p_c = z_0p_c - 1/2 * (self.dt/self.p)**2 * (w_c + APz_0p_c)
        z_1p_f = z_0p_f - 1/2 * (self.dt/self.p)**2 * (w_f + APz_0p_f)

        z_m1p_c = z_0p_c
        z_m1p_f = z_0p_f

        z_mp_c = z_1p_c
        z_mp_f = z_1p_f

        for _ in range(2, self.p + 1): # we need to perform steps m = 2 to p
            temp_c = z_mp_c.copy()
            temp_f = z_mp_f.copy()

            APz_mp_c = self.A_cf @ z_mp_f
            APz_mp_f = self.A_ff @ z_mp_f

            z_mp_c = 2*z_mp_c - z_m1p_c - (self.dt/self.p)**2 * (w_c + APz_mp_c)
            z_mp_f = 2*z_mp_f - z_m1p_f - (self.dt/self.p)**2 * (w_f + APz_mp_f)

            z_m1p_c = temp_c
            z_m1p_f = temp_f

        # Final update
        temp_c = self.z_c.copy()
        temp_f = self.z_f.copy()

        self.z_c = -self.z_previous_c + 2 * z_mp_c
        self.z_f = -self.z_previous_f + 2 * z_mp_f

        self.z_previous_c = temp_c
        self.z_previous_f = temp_f

        if update_u: # we need to reorganize the solution vector
            self.u[self.coarse_idx] = self.z_c # u is currently z
            self.u[self.fine_idx] = self.z_f # u is currently z
            self.u = self.M_bar_sqrt_inv @ self.u # now u is u

            self.u_previous[self.coarse_idx] = self.z_previous_c # u_previous is currently z_previous
            self.u_previous[self.fine_idx] = self.z_previous_f # u_previous is currently z_previous
            self.u_previous = self.M_bar_sqrt_inv @ self.u_previous

    def solve(self, T, t_eval=None):
        """
        TODO
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

        for i in tqdm(range(1, n_steps), desc="Solving the local time stepping Leap-Frog scheme"):
            # check if we need to store the solution that is if we are
            # at a time in t_eval or if we are such that previous step < t_eval < current step
            if i == n_steps - 1 or (len(t_eval) >= 2 and t_eval[0] <= i * self.dt < t_eval[1]):
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
        TODO
        """

        return self.fem.get_solution(u, x)

# == Plotting functions ==

def compare_time_evolution(problemSolver_as_a_function_of_dt, T, t_eval, dt_list):
    fig, ax = plt.subplots(len(t_eval), len(dt_list), figsize=(8, 30), sharex=True, sharey=True)
    for i, dt in enumerate(dt_list):
        print(f"Computing solution for dt = {dt}...")
        problem = problemSolver_as_a_function_of_dt(dt)
        solution = problem.solve(T, t_eval)

        for j, u in enumerate(solution):
            ax[j, i].plot([node.x for node in problem.fem.nodes], u, marker="o", markersize=5, linewidth=2)
            ax[j, i].set_xlim(0, problem.L)
            ax[j, i].set_ylim(-0.1, 1.5)
            ax[j, i].xaxis.set_tick_params(labelsize=15)
            ax[j, i].yaxis.set_tick_params(labelsize=15)
            #if i == 0:
            #    ax[j, i].set_ylabel(r"$u$", fontsize=20)
            ax[j,i].set_yticks([])
            if j == len(t_eval) - 1:
                ax[j, i].set_xlabel(r"$x$", fontsize=20)
            if j == 0:
                ax[j, i].set_title(r"$\Delta t = $" + f"{dt:.3f}", fontsize=20)
            # put the outeredge of the subplot in black
            for spine in ax[j, i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

    # Adjust the layout to make space for the arrow on the right
    fig.subplots_adjust(right=0.9)

    # Add a vertical time arrow on the right
    arrow_start_y = 0.875  # Start of arrow in figure coordinates
    arrow_end_y = 0.05  # End of arrow in figure coordinates
    arrow_x = 0.92  # X position of the arrow in figure coordinates

    # Add the arrow
    arrow = FancyArrow(
        x=arrow_x, y=arrow_start_y, dx=0, dy=arrow_end_y - arrow_start_y,
        width=0.001, length_includes_head=True, head_width=0.004, head_length=0.015,
        color="black", transform=fig.transFigure, clip_on=False
    )
    fig.patches.append(arrow)

    # Add ticks and labels along the arrow : each axes from the right column 
    for i, axis in enumerate(ax[:, -1]):
        pos = axis.get_position()  # Get the position of each axis
        tick_y = pos.y0 + (pos.y1 - pos.y0) / 2  # Vertical center of the axis
        fig.text(arrow_x + 0.005, tick_y, f"{t_eval[i]:.1f}", va="center", ha="left", fontsize=15)
        # put a tick on the arrow
        #fig.text(arrow_x, tick_y, "-", va="center", ha="center", fontsize=25)
        
        # Add horizontal line across the row
        start_line = ax[i, 0].get_position().x0
        end_line = arrow_x + 0.0025
        fig.add_artist(plt.Line2D(
            [start_line, end_line],  # X-coordinates of the line
            [tick_y, tick_y],  # Same y-coordinate as the tick
            color="black",  # Color of the line
            linewidth=1.5,  # Line width
            transform=fig.transFigure,
            zorder=0,  # Plot behind the subplots
            clip_on=False
        ))

    # Add a time label at the top of the arrow
    fig.text(arrow_x + 0.01, arrow_end_y + 0.02, "Time", ha="left", fontsize=16)
    plt.show()

if __name__ == "__main__":
    c = -1.0
    L = 4.0
    sigma = 0.4
    T = 90.0

    def u0(x, sigma=1., L=1.):
        return 1./(np.sqrt(2*np.pi)*sigma) * np.exp(-(x - L/2)**2/(2*sigma**2))

    def v0(x):
        return c
    
    if False:
        # == We define both the manual and classical FEM models ==
        # We generate a figure to show the mesh and the initial condition

        print("\n == Part 1 ==\n")

        h_coarse = 0.1
        n_elements_coarse = int(L/h_coarse)
        p = 4
        n_elements_refined = 2

        refined_mesh = []

        for i, x in enumerate(np.arange(0, L+h_coarse, h_coarse)):
            refined_mesh.append(x)
            if 1 <= x < 1 + h_coarse * n_elements_refined:
                for j in range(1, p):
                    refined_mesh.append(x + j * h_coarse / p)

        regular_mesh_problem = ClassicalFEM1DLeapFrog(n_elements_or_mesh=n_elements_coarse, dt=0.01, c=c, L=L, u0=lambda x: u0(x, sigma, L), v0=v0)
        refined_mesh_problem = ClassicalFEM1DLeapFrog(n_elements_or_mesh=refined_mesh, dt=0.01, c=c, L=L, u0=lambda x: u0(x, sigma, L), v0=v0)

        # plot of the initial condition (no title) and the computed u_previous
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        y_mesh = 0
        for ax_i, problem in zip(ax, [regular_mesh_problem, refined_mesh_problem]):
            ax_i.hlines(y_mesh, 0, L, colors='black', linestyles='solid')
            for node in problem.fem.nodes:
                ax_i.plot(node.x, y_mesh, 'k', marker='d', markersize=5)
                ax_i.vlines(node.x, y_mesh, problem.u[node.global_idx], colors='black', linestyles='dashed', alpha=0.5, linewidth=1.5)

            ax_i.set_xlim(0, L)
            ax_i.set_ylim(-0.1, 1.2)
            ax_i.xaxis.set_tick_params(labelsize=15)
            ax_i.yaxis.set_tick_params(labelsize=15)

        ax[0].plot([node.x for node in regular_mesh_problem.fem.nodes], regular_mesh_problem.u, marker="o", markersize=5, linewidth=2)
        ax[1].plot([node.x for node in refined_mesh_problem.fem.nodes], refined_mesh_problem.u, marker="o", markersize=5, linewidth=2)

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Find the min and max values for the color scale
        vmin = min(problem.A.min() for problem in [regular_mesh_problem, refined_mesh_problem])
        vmax = max(problem.A.max() for problem in [regular_mesh_problem, refined_mesh_problem])

        for ax, problem in zip(axs, [regular_mesh_problem, refined_mesh_problem]):
            im = ax.imshow(problem.A.toarray(), cmap=sns_color, vmin=vmin, vmax=vmax)
            ax.set_title(f"Mesh with {len(problem.fem.nodes)} nodes", fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Node index", fontsize=15)
            ax.set_ylabel("Node index", fontsize=15)

        # Add a common colorbar
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Matrix value', fontsize=15)
        plt.show()

        # == Question 2 : numerical exploration of the stability criterion on the coarse mesh ==
        # We compute the solution for different values of dt
        # The theoretical stability criterion is dt <= h_coarse / abs(c)
        # In our case, c = -1, so the stability criterion is dt <= h_coarse
        print("\n == Part 2 ==\n")

        dt_eval = 0.5
        t_eval = np.arange(0, T + dt_eval, dt_eval)
        dt_list = [h_coarse / 1.05, h_coarse, 1.05 * h_coarse]
        regular_problemSolver_as_a_function_of_dt = lambda dt: ClassicalFEM1DLeapFrog(n_elements_or_mesh=n_elements_coarse, dt=dt, c=c, L=L, u0=lambda x: u0(x, sigma, L), v0=v0)
        compare_time_evolution(regular_problemSolver_as_a_function_of_dt, T, t_eval, dt_list)

        # == Question 3 : numerical exploration of the stability criterion on the refined mesh ==
        # We compute the solution for different values of dt
        # The theoretical stability criterion is dt <= h_fine / abs(c)
        # In our case, c = -1, so the stability criterion is dt <= h_fine
        print("\n == Part 3 ==\n")

        h_fine = h_coarse / p
        dt_list = [h_fine / 1.05, h_fine, 1.05 * h_fine]
        refined_problemSolver_as_a_function_of_dt = lambda dt: ClassicalFEM1DLeapFrog(n_elements_or_mesh=refined_mesh, dt=dt, c=c, L=L, u0=lambda x: u0(x, sigma, L), v0=v0)
        compare_time_evolution(refined_problemSolver_as_a_function_of_dt, T, t_eval, dt_list)

        # == Question 4 : Local time stepping ==
        # We implement the local time stepping method

    print("\n == Part 4 ==\n")

    # We define the mesh and the mask for the fine nodes

    h_coarse = 0.1
    n_elements_coarse = int(L/h_coarse)
    p = 10
    n_elements_refined = 10

    refined_mesh = []
    mask_is_fine_node = []

    flag = False
    for i, x in enumerate(np.arange(0, L+h_coarse, h_coarse)):
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

    # Make a plot of the mesh with color coding for fine nodes
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(refined_mesh[mask_is_fine_node], np.zeros(len(refined_mesh))[mask_is_fine_node], 'ro', markersize=10, alpha=0.5, label="Fine node")
    ax.plot(refined_mesh[~mask_is_fine_node], np.zeros(len(refined_mesh))[~mask_is_fine_node], 'bo', markersize=10, alpha=0.5, label="Coarse node")

    ax.set_xlim(0, L)
    ax.set_ylim(-0.1, 0.1)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_yticks([])

    ax.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.subplots_adjust(bottom=0.4)
    plt.show()

    # We define the local time stepping problem
    problem = LocalTimeSteppingFEM1DLeapFrog(refined_mesh, mask_is_fine_node, dt=0.005*p, p=p, c=c, u0=lambda x: u0(x, sigma, L), v0=v0)
    problem2 = ClassicalFEM1DLeapFrog(n_elements_or_mesh=refined_mesh, dt=0.005, c=c, L=L, u0=lambda x: u0(x, sigma, L), v0=v0)
    # first do a static plot of u and u_previous
    fig, ax = plt.subplots(figsize=(10, 5))

    line, = ax.plot([node.x for node in problem.fem.nodes], problem.u, marker="o", markersize=5, linewidth=2)
    line_previous, = ax.plot([node.x for node in problem.fem.nodes], problem.u_previous, marker="o", markersize=5, linewidth=2)
    line2, = ax.plot([node.x for node in problem2.fem.nodes], problem2.u + 1, marker="o", markersize=5, linewidth=2)
    line2_previous, = ax.plot([node.x for node in problem2.fem.nodes], problem2.u_previous + 1, marker="o", markersize=5, linewidth=2)

    ax.set_xlim(0, L)
    ax.set_ylim(-0.1, 2.5)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$u$", fontsize=20)

    plt.legend(["$u$", "$u_{previous}$", "$u_{problem2}$", "$u_{previous, problem2}$"], fontsize=15)
    plt.show()

    # plot all the matrices
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    im = axs[0, 0].imshow(problem.A_cc.toarray(), cmap=sns_color)
    axs[0, 0].set_title("A_cc", fontsize=15)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xlabel("Coarse node index", fontsize=15)
    axs[0, 0].set_ylabel("Coarse node index", fontsize=15)

    im = axs[0, 1].imshow(problem.A_cf.toarray(), cmap=sns_color)
    axs[0, 1].set_title("A_cf", fontsize=15)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xlabel("Fine node index", fontsize=15)
    axs[0, 1].set_ylabel("Coarse node index", fontsize=15)

    im = axs[1, 0].imshow(problem.A_fc.toarray(), cmap=sns_color)
    axs[1, 0].set_title("A_fc", fontsize=15)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xlabel("Coarse node index", fontsize=15)
    axs[1, 0].set_ylabel("Fine node index", fontsize=15)

    im = axs[1, 1].imshow(problem.A_ff.toarray(), cmap=sns_color)
    axs[1, 1].set_title("A_ff", fontsize=15)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_xlabel("Fine node index", fontsize=15)
    axs[1, 1].set_ylabel("Fine node index", fontsize=15)

    # Add a common colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Matrix value', fontsize=15)
    plt.show()

    # We compute the solution
    solution = problem.solve(T, t_eval = np.arange(0, T, 0.1))
    solution2 = problem2.solve(T, t_eval = np.arange(0, T, 0.1))

    plt.show()
    # make an animation with the difference between both solutions
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    line, = ax[0].plot([node.x for node in problem.fem.nodes], solution[0], marker="o", markersize=5, linewidth=2)
    line2, = ax[0].plot([node.x for node in problem2.fem.nodes], solution2[0] + 1, marker="o", markersize=5, linewidth=2)
    line_diff, = ax[1].plot([node.x for node in problem.fem.nodes], solution[0] - solution2[0], marker="o", markersize=5, linewidth=2)

    ax[0].set_xlim(0, L)
    ax[0].set_ylim(-0.1, 2.5)
    ax[0].xaxis.set_tick_params(labelsize=15)
    ax[0].yaxis.set_tick_params(labelsize=15)
    ax[0].set_xlabel(r"$x$", fontsize=20)
    ax[0].set_ylabel(r"$u$", fontsize=20)

    ax[1].set_xlim(0, L)
    ax[1].set_ylim(-0.1, 0.1)
    ax[1].xaxis.set_tick_params(labelsize=15)
    ax[1].yaxis.set_tick_params(labelsize=15)
    ax[1].set_xlabel(r"$x$", fontsize=20)
    ax[1].set_ylabel(r"$u_{diff}$", fontsize=20)

    def update(frame):
        line.set_ydata(solution[frame])
        line2.set_ydata(solution2[frame])
        line_diff.set_ydata(solution[frame] - solution2[frame])
        return line, line2, line_diff

    anim = FuncAnimation(fig, update, frames=len(solution), blit=True, interval=50)
    plt.show()