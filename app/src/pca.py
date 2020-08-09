import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d


def run_pca(dat):
    """
    run pca on each brain area

    Args:
    X (numpy array of floats): Data matrix each column corresponds to a
                               different random variable

    Returns:
    variance_explained (numpy array of floats)  : percentage of variances explained by pcs
    V (numpy array of floats)  : Vector of eigenvalues
    W (numpy array of floats)  : Corresponding matrix of eigenvectors

    """
    dt = 10  # binning at 10 ms
    NT = dat.shape[-1]

    NN = len(dat)

    # top PC directions from stimulus + response period

    droll = np.reshape(dat, (NN, -1))  # first 80 bins = 1.6 sec
    droll = droll - np.mean(droll, axis=1)[:, np.newaxis]
    model = PCA(n_components=min(droll.shape[0], droll.shape[1])).fit(droll.T)

    W = model.components_  # eigenvectors
    V = model.explained_variance_  # eigenvalues
    csum = np.cumsum(V)
    variance_explained = csum / np.sum(V)

    return W, V, variance_explained


def smt_pca(dat, n):
    ## smoothing each individual trial and neuron
    # pc_10ms_smt = np.zeros((pc_10ms.shape[0],340,800))
    # x = np.linspace(0, dat.shape[-1]-1, num = dat.shape[-1])
    # xnew = np.linspace(0, np.max(x), num=800, endpoint=True)
    # f = interp1d(x, pc_10ms[0], axis = 1, kind='cubic')
    # pc_10ms_smt[0] = f(xnew)
    # print(pc_10ms_smt.shape)

    # smoothing mean trajectory

    x = np.linspace(0, len(dat) - 1, num=len(dat))
    xnew = np.linspace(0, np.max(x), num=n, endpoint=True)
    f = interp1d(x, dat, kind="cubic")
    pc_smt = f(xnew)

    return pc_smt


def map_pca(W, V, dat):
    """
    project entire trial data onto pcs

    Args:
    W (numpy array of floats): Data matrix each column corresponds to a
                               different random variable

    Returns:
    pc_10ms (numpy array of floats)  : projected neuron data on pcs; # of pc * # of bin

    """
    NN = len(dat)
    NT = dat.shape[-1]
    pc_10ms = W @ np.reshape(dat, (NN, -1))
    pc_10ms = np.reshape(pc_10ms, (len(V), -1, NT))
    return pc_10ms
