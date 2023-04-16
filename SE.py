import math
import numpy as np
from scipy.stats import norm


def coarse_graining(org_signal, scale):
    """Coarse-graining the signals.
    Args:
        org_signal: original signal,
        scale: desired scale
    Return:
        new_signal: coarse-grained signal
    """
    new_length = int(np.fix(len(org_signal) / scale))
    new_signal = np.zeros(new_length)
    for i in range(new_length):
        new_signal[i] = np.mean(org_signal[i * scale:(i + 1) * scale])

    return new_signal


def cal_fluctuation_dispersion_entropy(x, d, m, c):
    """Calcuate the dispersion entropy of given signal
    # NOTE: https://arxiv.org/pdf/1902.10825.pdf
    Args:
        :param x: signals (1-dim list or array)
        :param d: delay
        :param m: embedding dimension
        :param c: the number of classes
    Return:
        the dispersion entropy of x
    """
    y = norm.cdf(x, loc=np.mean(x), scale=np.std(x))
    z = np.round(c*y+0.5)

    num_dispersion_pattern = len(x) - (m - 1) * d
    pattern_set = {}
    for i in range(num_dispersion_pattern):
        pattern = ','.join([str(int(z[i:i+m][j]-np.min(z[i:i+m]))) for j in range(len(z[i:i+m]))])

        if pattern in pattern_set:
            pattern_set[pattern] += 1
        else:
            pattern_set[pattern] = 1

    fluctuation_dispersion_entropy = 0
    for key, value in pattern_set.items():
        prob = value / num_dispersion_pattern
        fluctuation_dispersion_entropy -= (prob) * math.log(prob)

    return fluctuation_dispersion_entropy


def multiscale_fluctuation_based_dispersion_entropy(signal, maxscale=20, classes=6, emb_dim=3, delay=1):
    """ Calculate multiscale fluctuation_based dispersion entropy.
    # NOTE: https://arxiv.org/pdf/1902.10825.pdf
    Args:
        :param signal: input signal,
        :param scale: coarse graining scale,
        :param classes: number of classes,
        :param emd_dim: embedding dimension,
        :param delay: time delay
    Return:
        mde: multiscale dispersion entropy value of the signal (list of float)
    """
    mfde = np.zeros(maxscale)
    for i in range(maxscale):
        cg_signal = coarse_graining(signal, i+1)
        en = cal_fluctuation_dispersion_entropy(cg_signal, d=delay, m=emb_dim, c=classes)
        mfde[i] = en

    return mfde

