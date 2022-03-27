import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.stats
import scipy.special
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp


def lv_fixed_points(s, c, regular=True, gamma=1, alpha=1.1, std_a=0,
                    a=None, sparse_mat=True, ic_num=10, runtime=1e3, migration=1e-10,
                    threshold=1e-3, min_threshold_scan=1e-5, max_threshold_scan=1e-2):
    """ The attributes of the equilibria of the LV equations found by solving the ODE with random initial conditions

    Parameters:
        s (int): The number of species in the pool
        c (int): The average number of interaction for each species (exact for random regular graphs)
        a (numpy array): The interaction matrix. If None, construct according to other parameters
        regular (bool): If True, use a random regular graph, otherwise erdos-renyi
        sparse_mat (bool): If true, use sparse form of matrix for LV simulations
        alpha (float): The average interaction strength
        std_a (float): The standard deviation of interaction strengths
        gamma (float): Correlation of the interactions a[i, j] and a[j, i]
        ic_num (int): The number of different initial conditions to solve the ODE with
        runtime (float): The runtime for the ODE solver
        migration (float): The species migration rate (same for all species)
        threshold (float): The threshold abundance above which we determine that a species is persistent.
        min_threshold_scan (float): If by taking 'threshold' a feasible, stable and uninvadable equilibrium isn't found,
                                    try equilibria using a minimum threshold 'min_threshold_scan'
        max_threshold_scan (float): If by taking 'threshold' a feasible, stable and uninvadable equilibrium isn't found,
                                    try equilibria using a maximum threshold 'max_threshold_scan'
    Returns:
        diversity (numpy array): Histogram of the diversity of the equilibria (The number of persistent species)
        largest_component (numpy array): The size of the largest connected component in each equilibrium found
        eq_not_found (int): The number of runs where no feasible, stable and uninvadable fixed point was found
        success (bool): whether the ODE solver succeeded in running
    """

    if a is None:  # If no interaction matrix was given generate a new one
        a = create_interaction_matrix(s, c, regular=regular, gamma=gamma, alpha=alpha, std_a=std_a)
    adjacency_mat = np.copy(a) - np.eye(s)  # The adjacency matrix
    adjacency_mat[adjacency_mat != 0] = 1
    diversity = np.zeros(s)  # Empty numpy array, to fill out
    largest_component = np.full(ic_num, np.nan)  # Empty numpy array, to fill out
    persistent, success = find_fps(a, ic_num=ic_num, runtime=runtime, migration=migration, sparse_mat=sparse_mat,
                                   threshold=threshold, min_threshold_scan=min_threshold_scan,
                                   max_threshold_scan=max_threshold_scan)
    # persistent - boolean matrix with the indices of the persistent species of each run; if no FP is found in a run,
    # the corresponding row will be all False
    if success:
        fp_run_idxs = np.any(persistent, axis=0)  # Indices of runs where a fixed point was found. Almost all runs reach
        # an equilibrium within the time frame we use in the paper, and so discarding the runs where this does not
        # occur is negligible
        eq_not_found = ic_num - np.sum(fp_run_idxs)
        if np.sum(fp_run_idxs) > 0:
            persistent = persistent[:, fp_run_idxs]  # Remove the runs where a fixed point wasn't found
            for fp_idx in range(np.sum(fp_run_idxs)):
                # For each fixed point, get diversity and largest component
                this_diversity = np.sum(persistent[:, fp_idx])
                diversity[this_diversity - 1] += 1
                # Get the networkx graph object for the matrix reduced to the subset of persistent species, then
                # find the largest connected component
                g = nx.from_numpy_array(adjacency_mat[np.ix_(persistent[:, fp_idx], persistent[:, fp_idx])])
                largest_component[fp_idx] = len(max(nx.connected_components(g), key=len))
    else:
        eq_not_found = None
    return diversity, largest_component, eq_not_found, success


def create_interaction_matrix(s, c, adjacency_mat=None, regular=True, gamma=1, alpha=1, std_a=0):
    """ Creates an interaction matrix

    Parameters:
         s (int): Interaction matrix size
         c (int): Average edges per vertex
         adjacency_mat (numpy array): The adjacency matrix. If None, a new one will be created
         regular (bool): If true use random regular graph, if false erdos-renyi
         gamma (float): The correlation of a[i, j] and a[j, i]
         alpha (float): The average of the non-zero entries
         std_a (float): The standard deviation of the non-zero entries

    Returns:
        a (numpy array): The interaction matrix
    """

    if adjacency_mat is None:
        # If no adjacency matrix was given, create a new one
        if regular:
            if c <= 0.5 * s:
                adjacency_mat = nx.to_numpy_array(nx.random_regular_graph(c, s))
            else:
                # Since nx.random_regular_graph is much more efficient for small c, if c is larger than s/2 get the
                # complementary adjacency matrix and invert
                adjacency_mat = nx.to_numpy_array(nx.random_regular_graph(s - c - 1, s))
                adjacency_mat = 1 - adjacency_mat - np.eye(s)
        else:
            adjacency_mat = nx.to_numpy_array(nx.erdos_renyi_graph(s, c / s))
    # Generate a full, correlated interaction matrix: create an SxS matrix of pairs of values with correlation gamma,
    # then use the pairs from the upper triangular matrix to populate the entire interactions matrix
    a = np.random.multivariate_normal(alpha * np.ones(2),
                                      (std_a ** 2) * (np.eye(2) + gamma * np.fliplr(np.eye(2))), (s, s))
    a = np.triu(a[:, :, 0], 1) + np.transpose(np.triu(a[:, :, 1], 1))
    a = np.multiply(a, adjacency_mat)  # Keep only interactions from adjacency_mat
    a += np.eye(s)  # Add self-interactions

    return a


def lv_ode_solver(runtime, a, ic_num=None, migration=1e-10, n0=None, sparse_mat=True,
                  rtol=1e-12, atol=1e-12, method='RK45'):
    """Runs noiseless LV time evolution.

        Parameters:
            runtime (float): Runtime for the solver
            a (numpy array): Interaction matrix
            ic_num (int): Number of different initial conditions to solve for
            migration (float): Species migration rate
            n0 (numpy array): Initial populations sizes, given as a 2d S x ic_num array. If None, draw randomly from
                              uniform [0, 1] distribution
            sparse_mat (bool): If true, use sparse form of matrix for LV simulations
            rtol (float): ODE solver relative tolerance
            atol (float): ODE solver absolute tolerance
            method (str): Integration method to be used

        Returns:
            t (numpy array): Calculation times
            y (numpy array): Population numbers at the calculation times
        """

    s = int(np.shape(a)[0])
    if n0 is None:
        # If no initial conditions are given, draw at random
        n0 = np.random.rand(s * ic_num)
    else:
        # Get the number of different initial conditions given
        if ic_num is None:
            if len(np.shape(n0)) == 1:
                ic_num = 1
            else:
                ic_num = np.shape(n0)[-1]
        n0 = np.reshape(n0, s * ic_num, order='F')  # n0 is reshaped because solve_ivp can only receive 1d vectors as
                                                    # initial conditions
    if sparse_mat and type(a) != scipy.sparse.csr_matrix:
        lv_a = csr_matrix(a)
    else:
        lv_a = a.copy()
    if ic_num == 1:
        def lv_func(t, func_n):
            return func_n * (1 - lv_a.dot(func_n)) + migration
    else:
        def lv_func(t, func_n):
            return np.reshape(np.reshape(func_n, (s, ic_num), order='F') * (
                    1 - lv_a.dot(np.reshape(func_n, (s, ic_num), order='F'))) + migration,
                              s * ic_num, order='F')
    sol = solve_ivp(lv_func, (0, runtime), n0, t_eval=np.array([0, runtime]), rtol=rtol, atol=atol, method=method)
    t = sol.t[-1]
    y = np.reshape(np.transpose(sol.y[:, -1]), (s, ic_num), order='F')

    if not sol.success:
        t = y = None

    return t, y


def find_fps(a, ic_num=1, runtime=5e4, migration=1e-10, sparse_mat=True,
             threshold=1e-3, min_threshold_scan=1e-5, max_threshold_scan=1e-2):
    """ Returns parameters of a single fixed point for the interaction matrix a

    Parameters:
        a (numpy array): The interaction matrix
        ic_num (int): The number of different initial conditions to run
        runtime (float): The maximum runtime the LV simulation is allowed to run without finding a fixed point
        threshold (float): threshold below which a species is considered extinct
        min_threshold_scan (float): If at 'threshold' a fixed point isn't found, try fixed points with
                                    a minimum threshold 'min_threshold_scan'
        max_threshold_scan (float): If at 'threshold' a fixed point isn't found, try fixed points with
                                    a maximum threshold 'max_threshold_scan'
        sparse_mat (bool): If true, use sparse form of matrix for LV simulations
        migration (float): The species migration rate

    Returns:
        persistent (numpy array): Boolean array of persistent species
        success(bool): Whether the ODE solver ran succeeded

    """
    s = int(np.shape(a)[0])
    abundances = np.random.rand(s, ic_num)
    persistent = np.zeros((s, ic_num), dtype=bool)

    if sparse_mat and type(a) != scipy.sparse.csr_matrix:
        lv_a = csr_matrix(a)
    else:
        lv_a = a.copy()
    abundances = lv_ode_solver(runtime, lv_a, migration=migration,
                               n0=abundances, sparse_mat=sparse_mat)[1]
    if abundances is not None:
        success = True
        for ic_idx in range(ic_num):
            these_lv_abundances = abundances[:, ic_idx]
            # As sometime there are extinct species with abundances that decay very slowly, defining species with
            # abundance above the threshold as persistent might cause us to miss the FP. Therefore, if this simple
            # threshold does not yield a feasible, stable and uninvadable FP, we test other possible thresholds; this
            # is done by taking all the species with abundances above min_threshold_scan as persistent,
            # then discarding them one by one, from low to high abundances; we stop this process and assume the run did
            # not reach a FP yet if we do not find a fixed point by the time the lowest abundance of the species we are
            # testing is at max_threshold_scan.
            # Note that Lemma #2 in appendix 1 ensures that the set of persistent species in one equilibrium cannot
            # be a subset of the set of another equilibrium; therefore the process of assigning different thresholds
            # and taking all species with higher abundances to be persistent can only yield one equilibrium.

            # All possible thresholds, unique and sorted
            thresholds = np.unique(these_lv_abundances[np.logical_and(these_lv_abundances >= min_threshold_scan,
                                                                      these_lv_abundances <= max_threshold_scan)])
            if threshold is not None:
                # If a specific threshold value for persistent abundances is given, test it first
                thresholds = np.insert(thresholds, 0, threshold)
            thresh_idx = 0  # Index going over the values in 'thresholds'
            fsu_eq = False  # Flag that becomes True if a feasible, stable and uninvadable equilibrium has been found
            while not fsu_eq and thresh_idx < len(thresholds):
                # As long as a feasible stable and uninvadable FP wasn't found, go over all possible thresholds
                these_persistent = these_lv_abundances >= thresholds[thresh_idx]  # Indices of species with abundance
                                                                                  # above this threshold
                fsu_eq = subset_fsu(a, these_persistent)  # True if this subset of persistent species generates a
                                                          # feasible, stable, uninvadable equilibrium
                thresh_idx += 1
            if fsu_eq:
                persistent[:, ic_idx] = these_persistent
    else:
        persistent[:] = None
        success = False
    return persistent, success


def subset_fsu(a, subset):
    """Checks whether a subset of persistent species defines a feasible, stable and uninvadable equilibrium

    Parameters:
        a (numpy array): The interactions matrix
        subset (numpy array): Boolean vector of the subset to be checked

    Returns True if the subset yields a feasible, stable and uninvadable equilibrium; False otherwise
    """

    a_star = a[np.ix_(subset, subset)]  # The matrix reduced to the space of the subset of persistent species
    if np.linalg.det(a_star) == 0:  # If the matrix is singular, return false
        return False
    else:
        subset_size = np.sum(subset)
        persistent_abundances = np.dot(np.linalg.inv(a_star), np.ones((subset_size, 1)))  # Abundances of the
                                                                                          # persistent species
        eigs = np.real(np.linalg.eigvals(np.dot(np.diag(np.reshape(persistent_abundances, subset_size)),
                                                          a_star)))
        feasibility = np.min(persistent_abundances) > 0
        stability = np.min(eigs) > 0
        uninvadability = np.all(np.dot(a[np.ix_(np.logical_not(subset), subset)], persistent_abundances) >= 1)

    return feasibility and stability and uninvadability


def main():
    # Recreates either figure 1 or figure 2E from the paper
    s = 400
    c = 3
    runs = 10  # For most results in the paper, we use runs = 30
    ic_num = 5  # For most results in the paper, we use ic_num = 15
    runtime = 1e4  # For most results in the paper, we use runtime = 5e4
    regular = True
    fig = ''
    while fig != '1' and fig != '2':
        fig = input('Would you like to recreate Fig. 1 (about 20 minutes) or Fig. 2E (about 5 minutes)? (1/2)')
        if fig == '1':
            alpha = np.array([0, 0.2, 0.325, 0.353, 0.36, 0.375, 0.4, 0.425, 0.45, 0.475, 0.485, 0.515, 0.51,
                              0.53, 0.54, 0.554, 0.556, 0.6, 0.61, 0.617, 0.619, 0.75, 0.95, 0.99, 1.01, 1.1, 1.2])
        elif fig == '2':
            alpha = np.array([0.985, 0.99, 0.995, 1.005, 1.01, 1.015]) * 0.5 / np.cos(np.pi / 7)
    diversity = np.zeros((len(alpha), runs, s))
    phi = np.zeros(len(alpha))
    phi_err = np.zeros(len(alpha))
    for k, aa in enumerate(alpha):
        for run in range(runs):
            res = lv_fixed_points(s, c, regular=regular, alpha=aa, ic_num=ic_num, runtime=runtime)
            diversity[k, run, :] = res[0]
        this_diversity_sum = np.sum(diversity[k, :, :], 0)
        phi[k] = np.dot(this_diversity_sum, np.arange(1, s + 1) / s) / np.sum(this_diversity_sum)
        phi_sqr = np.dot(this_diversity_sum, np.power(np.arange(1, s + 1) / s, 2)) / np.sum(this_diversity_sum)
        phi_err[k] = np.sqrt(phi_sqr - phi[k]**2)/np.sqrt(np.sum(this_diversity_sum))
    plt.errorbar(alpha, phi, phi_err)
    plt.xlabel('alpha')
    plt.ylabel('phi')
    plt.title('Random regular graph, S=400 C=3')

    plt.show()


if __name__ == "__main__":
    main()
