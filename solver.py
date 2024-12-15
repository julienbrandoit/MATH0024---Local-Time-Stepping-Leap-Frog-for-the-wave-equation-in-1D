
#%% IMPORTS %%#

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix as smat


#%% CONSTANTS %%#

plt.rcParams.update({'font.size': 10, 'axes.grid': True})

M_ELEM = np.array([
    [1/3, 1/6],
    [1/6, 1/3]
])

K_ELEM = np.array([
    [1, -1],
    [-1, 1]
])


#%% ROUTINES %%#

def routine(spatial_tab, time_tab, speed, u_analytical,
            refinement, n_images, speedup, leapfrog):
    
    time_step           = time_tab[1] - time_tab[0]
    spatial_step_tab    = np.diff(spatial_tab)
    convergence         = abs(speed) * time_step / spatial_step_tab
    
    if leapfrog:
        print(f"CONVERGENCE FACTOR          :         {min(convergence)}")
        if min(convergence) > 1 :
            print(
            "======== WARNING : convergence criterion is not verified ========"
            )
    else:
        print(f"CONVERGENCE FACTOR          :         {max(convergence)}")
        if max(convergence) > 1 :
            print(
            "======== WARNING : convergence criterion is not verified ========"
            )

    n_nodes = len(spatial_tab)
    M_glob  = np.zeros((n_nodes,))
    K_glob  = np.zeros((n_nodes, n_nodes))

    # Assembly of the global K and M matrices
    for i in range(n_nodes-1):
        M_glob[i:i+2]           += np.sum(M_ELEM, axis=1) / convergence[i]
        K_glob[i:i+2, i:i+2]    += K_ELEM * convergence[i]

    M_glob_inv  = np.diag(M_glob**(-1))
    A_glob      = smat(np.dot(M_glob_inv, K_glob))
    
    update_fine     = smat([])
    update_coarse   = smat([])
    
    if leapfrog:
        eye             = smat(np.eye(A_glob.shape[0]))
        localization    = get_localization(spatial_tab)
        update_fine     = A_glob.dot(localization)
        update_coarse   = A_glob.dot(eye - localization)
    
    u       = np.empty((len(time_tab)+1, n_nodes))
    u[0]    = u_analytical(-time_step)
    u[1]    = u_analytical(0)
    
    frames          = []
    refresh_rate    = 0
    index_tab       = range(len(time_tab))
    
    if n_images != 0:
        refresh_rate    = int(len(time_tab)/n_images)
        index_tab       = tqdm(index_tab)
        
    for i in index_tab:
        
        # add a frame to the gif
        if n_images != 0 and (time_tab[i]/time_step)%refresh_rate == 0:
            
            plt.plot(spatial_tab, u[i+1], "-o", label="Numerical solution")
            plt.plot(spatial_tab, u_analytical(time_tab[i]),
                     label="Analytical solution (without reflection)")
            plt.xlabel(r'Independent variable $x$ [-]')
            plt.ylabel(r'Solution $u(x,t)$ [-]')
            plt.title(f'Time: $t =$ {round(time_tab[i], 3)} [s]')
            plt.ylim(-0.1, 1.3*max(u_analytical(0)))
            plt.xlim(spatial_tab[0], spatial_tab[-1])
            plt.legend(loc="upper right")
            
            filename = f'time_{time_tab[i]}.png'
            plt.savefig(filename)
            plt.close()
            frames.append(imageio.imread(filename))
            os.remove(filename)
            
        if i == len(time_tab)-1:
            break
        
        if leapfrog:
            
            w       = update_coarse.dot(u[i+1])
            utprev  = u[i+1]
            ut      = utprev - 1/2 * (1/refinement)**2                         \
                                * (w + update_fine.dot(utprev))
            
            for p in range(2, refinement+1):
                utnext  = 2*ut - utprev - (1/refinement)**2                    \
                                        * (w + update_fine.dot(ut))
                utprev  = ut
                ut      = utnext
            
            u[i+2]  = 2*ut - u[i]
        
        else:
            u[i+2]  = 2*u[i+1] - A_glob.dot(u[i+1]) - u[i]
    
    if n_images != 0:
        fps = n_images/time_tab[-1] * speedup
        imageio.mimsave('animation.gif', frames, fps=fps)
    
    return u
    
def get_spatial_tab(spatial_tab_coarse, fine_regions, refinement):
    
    space_tab_tot = spatial_tab_coarse
    
    # we append the position of the fine nodes
    for i in fine_regions:
        space_tab_add = np.linspace(spatial_tab_coarse[i],
                                    spatial_tab_coarse[i+1],
                                    refinement+1)[1:-1]
        
        space_tab_tot = np.append(space_tab_tot, space_tab_add)
        
    # we sort the list so that all the fine positions are not at the end
    space_tab_tot.sort()
    
    return space_tab_tot

def get_localization(space_tab):
    
    diff    = np.diff(space_tab)
    fine    = min(diff)
    
    # we divide by `fine` to improve the use of `round`
    dx_tab  = [np.inf]                                                         \
            + [round((space_tab[i]-space_tab[i-1])/fine)
                for i in range(1, len(space_tab))]                             \
            + [np.inf]
           
    # we store a 1 when the minimal spacing of a node relative to its direct
    # neighbors corresponds to a fine spacing, we store a 0 in the other case
    localization = [1 if min([dx_tab[i], dx_tab[i-1]]) == 1 else 0             \
                    for i in range(1, len(dx_tab))]
    
    return smat(np.diag(localization))

def plot_subplots(u, spatial_tab, time_tab, nrows=5):
    
    fig, ax = plt.subplots(nrows=nrows, sharex=True, sharey=True)
    
    for i in range(nrows):
        index = int((nrows-1-i)/time_tab[-1] * (len(u)-2))
        ax[i].plot(spatial_tab, u[index+1], "-o", alpha=0.5,
                    label=f"t = {round(time_tab[index])} [s]")
        ax[i].legend()
        
    plt.xlabel(r'Independent variable $x$ [-]')
    fig.supylabel(r'Solution $u(x,t)$ [-]', size=plt.rcParams["font.size"])
    plt.tight_layout()
    plt.show()
    
def plot_error(s_tab, refinement, fine_regions, error):
    
    plt.vlines(1, 1e-100, 1e+100, "k", linestyles="--", alpha=0.6,
                   label=r"Coarse criterion")
    
    if refinement != 0 and len(fine_regions) != 0:
        plt.vlines(1/refinement, 1e-100, 1e+100, "k", linestyles="dotted",
                   alpha=0.6, label=r"Fine criterion")
        
    plt.plot(s_tab, error[:len(s_tab)])
    plt.xlim(left=0.7e-2, right=1e1)
    plt.ylim(bottom=0.8, top=1.5)
    plt.xscale('log')
    plt.xlabel(r'Convergence factor $s$ [-]')
    plt.ylabel(r'Error estimator $\epsilon$ [-]')
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.show()
    
    
#%% MAIN %%#

if __name__ == "__main__":
    
    
    ################################################################
    #     ONLY PARAMETERS NEEDED TO OBTAIN THE REPORT FIGURES      #
    #     mode      : either "gif", "subplots" or "error"          #
    #     with_fine : set to True to add fine elements             #
    #     leapfrog  : in the case of fine elements, whether to     #
    #                   solve using LTS-LF                         #
    
    MODE        = "subplots"
    WITH_FINE   = False
    LEAPFROG    = WITH_FINE and True    #! only change what follows the `and`
    
    #                                                              #
    ################################################################
    
    
    DT_TAB  = np.geomspace(1e-3, 1, 100)         if MODE == "error"            \
         else np.array([1e-1])                   if MODE == "gif"              \
         else np.array([0.9e-1, 1e-1, 1.1e-1])/3 if WITH_FINE and not LEAPFROG \
         else np.array([0.9e-1, 1e-1, 1.1e-1])
    
    SPATIAL_SPAN    = 4
    SPATIAL_STEP_C  = 0.1
    N_NODES_C       = int(SPATIAL_SPAN/SPATIAL_STEP_C) + 1
    SPATIAL_TAB_C   = np.linspace(0, SPATIAL_SPAN, N_NODES_C)

    TIME_SPAN       = 9
    
    FINE_REGIONS    = [N_NODES_C//4, N_NODES_C//4+1] if WITH_FINE else []
    REFINEMENT      = 3 if WITH_FINE != 0 else 1
    SPATIAL_TAB     = get_spatial_tab(SPATIAL_TAB_C, FINE_REGIONS, REFINEMENT)
    N_NODES         = len(SPATIAL_TAB)
    
    SPEED           = -1
    SIGMA           = 0.4
    X_BAR           = SPATIAL_SPAN/2
    
    def U_ANALYTICAL(t):
        arg = SPATIAL_TAB - np.array([SPEED*t + X_BAR]*N_NODES)
        return 1/np.sqrt(2*np.pi)/SIGMA * np.exp(-arg**2/(2*SIGMA**2))

    N_IMAGES    = min(TIME_SPAN/min(DT_TAB), 100) if MODE == "gif" else 0       
    SPEEDUP     = 1
    
    print(f"""
CONSTANT PARAMETERS
------------------------------

CONFIGURATION MODE          :       {MODE}
HAS FINE REGIONS            :       {WITH_FINE}
USE OF LEAPFROG             :       {LEAPFROG}

TIME SPAN                   :       {TIME_SPAN}

SPATIAL SPAN                :       {SPATIAL_SPAN}
SPATIAL STEP (COARSE)       :       {SPATIAL_STEP_C}
NUMBER OF NODES (COARSE)    :       {N_NODES_C}

INDICES OF FINE REGIONS     :       {FINE_REGIONS}
REFINEMENT                  :       {REFINEMENT}
TOTAL NUMBER OF NODES       :       {N_NODES}

INITIAL POSITION            :       {X_BAR}
WAVE SPEED                  :       {SPEED}
STANDARD DEVIATION          :       {SIGMA}

------------------------------
""")
    
    error   = np.zeros_like(DT_TAB)
    i       = 0
    
    it = enumerate(tqdm(DT_TAB)) if MODE == "error" else enumerate(DT_TAB)
    for i, dt in it:
        
        N_TICKS         = int(TIME_SPAN/dt +1)
        TIME_TAB        = np.linspace(0, TIME_SPAN, N_TICKS)
        
        print("\nCOMPUTING ...")
        
        u = routine(SPATIAL_TAB, TIME_TAB, SPEED, U_ANALYTICAL,
                    REFINEMENT, N_IMAGES, SPEEDUP, leapfrog=LEAPFROG)
        
        if MODE == "subplots":
            plot_subplots(u, SPATIAL_TAB, TIME_TAB)            
            
        if MODE == "error":
            error[i] = max(u[-1])
            if error[i] > 100 or np.isnan(error[i]):
                error[i] = 100
                print("\n\nDIVERGENCE OCCURRED, ERROR ANALYSIS STOPPED HERE")
                break
            
        print("")
    
    if MODE == "error":
        s_tab = abs(SPEED * DT_TAB / SPATIAL_STEP_C)
        plot_error(s_tab[:i+1], REFINEMENT, FINE_REGIONS, error)