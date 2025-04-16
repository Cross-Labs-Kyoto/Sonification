import pickle
from copy import deepcopy

import networkx as nx
import numpy as np
from scipy.stats import linregress, zscore, pearsonr, multivariate_normal, norm

LATTICE_ORIG = pickle.load(open("phi_lattice_22.pickle", "rb"))
DISTANCES = nx.shortest_path_length(LATTICE_ORIG, target=(((0,), (1,)), ((0,), (1,))))

ORDER = []
for distance in range(max(DISTANCES.values()) + 1):
    ORDER += [key for key in DISTANCES.keys() if DISTANCES[key] == distance]

PHIR_ATOMS = {  # Only used for the phi_r function.
    (((0,), (1,)), ((0, 1),)),
    (((1,),), ((0, 1),)),
    (((0, 1),), ((0,),)),
    (((0, 1),), ((0,), (1,))),
    (((0, 1),), ((1,),)),
    (((0, 1),), ((0, 1),)),
    (((0,),), ((1,),)),
    (((1,),), ((0,),)),
}


def corrected_zscore(data, axis=1, noise=10 ** -6):
    data = zscore(data, axis=axis)
    for i, row in enumerate(data):
        if all(np.isnan(row)):
            data[i] = np.random.normal(loc=0.0, scale=noise, size=len(row))
    return data


def local_entropy_1d(idx1, x):
    mu = x[idx1].mean()  # Central tendency
    sigma = x[idx1].std()  # Standard deviation
    # Magic call to scipy.
    entropy = -np.log(norm.pdf(x[idx1], loc=mu, scale=sigma))
    return entropy


def local_entropy_nd(x):
    # It gets grumpy if it's a 2D y with only one row
    # so in that case, we kick it to local_entropy_1d.
    if x.shape[0] == 1:
        return local_entropy_1d(0, x)
    else:
        cov = np.cov(x, ddof=0)  # The covariance matrix.
        means = x.mean(axis=-1)  # The central tendencies
        # Magic call to scipy.
        entropy = -np.log(multivariate_normal.pdf(x.T, mean=means, cov=cov))
    return entropy


def local_phi_min(idx1, idx2, atom, x, lag=1):
    n1 = x.shape[1]
    # This thing is called edge b/c of the earlier project I wrote it for.
    edge = x[[idx1, idx2], :]
    i_plus = np.repeat(np.inf, n1 - lag)  # Informative
    i_minus = np.repeat(np.inf, n1 - lag)  # Misinformative
    len_atom_0 = len(atom[0])
    len_atom_1 = len(atom[1])
    for i in range(len_atom_0):
        # For i(s;t) = h(s) - h(s|t), h(s) is the informative probability mass exclusion.
        # So the informative part of the MMI is just the entropy of the past process.
        edge_i = edge[((atom[0][i]),)][:, :-lag]  # The time series of the ith element
        h_edge_i = local_entropy_nd(edge_i)  # The entropy of edge i.
        # Minimum entropy as redundancy.
        i_plus = np.minimum(i_plus, h_edge_i)
        # The misinformative probability mass exclusions: h(s|t)
        # We are taking the min over all s and t.
        # h(s|t) = h(s,t) - h(t)
        for j in range(len_atom_1):
            # Joint-state of past and future.
            joint = np.squeeze(
                np.vstack((
                    edge[(atom[0][i],)][:, :-lag],
                    edge[(atom[1][j],)][:, lag:]
                ))
            )
            # Marginal future
            marginal = edge[(atom[1][j],)][:, lag:]
            # h(s|t) = h(s,t) - h(t)
            conditional = np.subtract(local_entropy_nd(joint),
                                      local_entropy_nd(marginal)
                                      )
            # Minimum redundancy
            i_minus = np.minimum(i_minus,
                                 conditional
                                 )
    # mi is informative minus misinformative.
    return np.subtract(i_plus, i_minus)


def local_phi_id(idx1, idx2, x):
    lattice = deepcopy(LATTICE_ORIG)
    # Going up the lattice in ascending order
    for atom in ORDER:
        # Compute the redundancy
        lattice.nodes[atom]["phi_min"] = local_phi_min(idx1, idx2, atom, x)
        # At the bottom, the redundancy is the partial information
        if atom == (((0,), (1,)), ((0,), (1,))):
            lattice.nodes[atom]["pi"] = lattice.nodes[atom]["phi_min"]
        else:  # Mobius inversion
            lattice.nodes[atom]["pi"] = np.subtract(lattice.nodes[atom]["phi_min"],
                                                    np.vstack(([lattice.nodes[a]["pi"] for a in
                                                                lattice.nodes[atom]["descendants"]])).sum(axis=0)
                                                    )
    return lattice


def local_phi_r(phi_lattice):
    # Phir is the sum of a subset of integrated information atoms
    phir = phi_lattice.nodes[(((0,),), ((0, 1),))]["pi"]
    for atom in PHIR_ATOMS:
        phir += phi_lattice.nodes[atom]["pi"]
    return phir


def remove_autocorrelation(x):
    n0 = x.shape[0]
    n1 = x.shape[1]
    regressed = np.zeros((n0, n1 - 1))
    for i in range(n0):  # Each row is regressed independently of all others.
        x_i = x[i].copy()
        # Computing the linear correlation between time {t-1} and time {t}
        lr = linregress(x_i[:-1], x_i[1:])
        # The predicted values at time {t} given the regression.
        y_pred = lr[1] + np.nanprod([np.repeat(lr[0], len(x_i[:-1])), x_i[:-1]], axis=0)
        # Computing the residuals.
        residuals = np.nansum([x_i[1:], -y_pred], axis=0)
        regressed[i, :] = residuals
    return corrected_zscore(regressed, axis=-1)


def global_signal_regression(x):
    n0 = x.shape[0]
    n1 = x.shape[1]
    gsr = np.zeros((n0, n1), dtype=np.float64)  # Initialize GSR y
    mean = np.nanmean(x, axis=0)  # Compute global signal
    for i in range(n0):
        lr = linregress(mean, x[i])  # Linregress each channel against the GS
        y_pred = lr[1] + (lr[0] * mean)
        z = np.nansum([x[i], -y_pred], axis=0)  # Regress out
        for j in range(n1):  # No need to iterate over columns, but it's fine.
            gsr[i, j] = z[j]  # From an earlier function in C.
    return corrected_zscore(gsr, axis=-1)


def mutual_information_matrix(x, alpha, lag=1, bonferonni=True):
    n0 = x.shape[0]
    mi_mat = np.zeros((n0, n0))
    # The bonferonni correction.
    if bonferonni:
        alpha_corr = alpha / (((n0 ** 2.0) - n0) / 2.0)
    else:
        alpha_corr = 1 * alpha
    for i in range(n0):
        for j in range(i):  # You only need to do the upper triangle.
            if lag == 0:
                r, p = pearsonr(x[i], x[j])
                if p < alpha_corr:  # No point computing a log if you don't have to...
                    mi = -0.5 * np.log(1 - (r ** 2.0))  # Gaussian MI from Pearson's r
                    mi_mat[i, j] = mi
                    mi_mat[j, i] = mi
            elif lag > 0:
                r1, p1 = pearsonr(x[i, :-lag], x[j, lag:])
                r2, p2 = pearsonr(x[i, lag:], x[j, :-lag])
                if p1 < alpha_corr:
                    mi1 = -0.5 * np.log(1.0 - (r1 ** 2.0))
                else:
                    mi1 = 0
                if p2 < alpha_corr:
                    mi2 = -0.5 * np.log(1.0 - (r2 ** 2.0))
                else:
                    mi2 = 0
                mi_mat[i, j] = mi1 + mi2
                mi_mat[j, i] = mi1 + mi2
    return np.array(mi_mat)


def minimum_information_bipartition(mi_mat, noise=False, noise_level=10 ** -6):
    n0 = mi_mat.shape[0]
    # The noise is a uniform floor of low mutual information connections.
    # This will ensure the graph is connected, but will slightly weaken
    # the strength of the modular struture.
    if noise:
        mi_mat_corr = np.add(mi_mat, noise_level)
    else:
        mi_mat_corr = 1 * mi_mat
    # Constructing the networkx object.
    g = nx.from_numpy_array(mi_mat_corr, create_using=nx.Graph())
    # Setting normalized to True increases the runtime dramatically.
    # I don't really know why.
    fiedler = nx.fiedler_vector(g, weight="weight", normalized=False)
    bipartition = [
        [i for i in range(n0) if fiedler[i] > 0],
        [i for i in range(n0) if fiedler[i] < 0]
    ]
    return bipartition


def local_total_correlation(x):
    n0 = x.shape[0]
    n1 = x.shape[1]
    # The joint entropy of the whole
    joint_ents = local_entropy_nd(x)
    # Pre-allocating space for the sum of the local marginal entropies
    sum_marginal_ents = np.zeros(n1, dtype=np.float64)
    for i in range(n0):
        marginal_ent = local_entropy_1d(i, x)
        sum_marginal_ents = np.add(sum_marginal_ents, marginal_ent)
    return np.subtract(sum_marginal_ents, joint_ents)


def local_o_information(x, local_tc=None):
    n0 = x.shape[0]
    n1 = x.shape[1]
    factor = (2.0 - n0)
    whole_tc = np.multiply(factor, local_total_correlation(x) if local_tc is None else local_tc)
    # Pre-allocating the sum of the local residual TCs
    sum_residual_tcs = np.zeros(n1, dtype=np.float64)
    # The local TC series of X_residual
    for i in range(n0):
        x_residuals = x[[j for j in range(n0) if j != i], :]
        residual_tc = local_total_correlation(x_residuals)
        sum_residual_tcs = np.add(sum_residual_tcs, residual_tc)
    return np.add(whole_tc, sum_residual_tcs)


def local_dual_total_correlation(x):
    return np.subtract(local_total_correlation(x), local_o_information(x))


def local_s_information(x):
    return np.add(local_total_correlation(x), local_dual_total_correlation(x))


def local_tse_complexity(x, num_samples=100):
    n0 = x.shape[0]
    n1 = x.shape[1]
    tc_whole = local_total_correlation(x)
    tse = np.zeros(n1, dtype=np.float64)
    for k in range(1, n0):
        factor = float(k) / n0
        null_tc = np.multiply(factor, tc_whole)
        sample_tcs_k = np.zeros(n1)
        for i in range(num_samples):
            if k > 1:
                choice = np.random.choice(n0, k, replace=False)
                x_choice = x[choice, :]
                sample_tcs_k = np.add(sample_tcs_k, local_total_correlation(x_choice))
        sample_tcs_k = np.divide(sample_tcs_k, float(num_samples))
        diff = np.subtract(null_tc, sample_tcs_k)
        tse = np.add(tse, diff)
    return tse
