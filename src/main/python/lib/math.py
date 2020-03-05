import multiprocessing

multiprocessing.freeze_support()

import matplotlib
import matplotlib.ticker
import sklearn.neighbors

import itertools
from ui.misc import ProgressBar
from typing import Union, Tuple, List
import scipy.signal
import numpy as np
import pandas as pd
import scipy.signal
import scipy.optimize
import scipy.stats
import sklearn.cluster
import sklearn.mixture
import hmmlearn.hmm
import scipy.stats
from scipy.special import expit

pd.options.mode.chained_assignment = None


def contains_nan(array):
    """
    Returns True if array contains nan values
    """
    return np.isnan(np.sum(array))


def count_n_states(class_probs):
    """
    Count number of states in trace, given propabilities

    Assumes the mapping
    class 4 -> 1 state
    class 5 -> 2 states
    etc...
    """
    classes_w_states = class_probs[[4, 5, 6, 7, 8]]
    return np.argmax(classes_w_states) + 1


def single_exp_fit(x, scale):
    """
    Single exponential fit.
    """
    return scipy.stats.expon.pdf(x, loc=0, scale=scale)


def leastsq_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Return the least-squares solution to a linear matrix equation.
    """
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def correct_DA(intensities, alpha=0, delta=0):
    """
    Calculates corrected Dexc-Aem intensity, for use in E and S calculations.
    """
    grn_int, grn_bg, acc_int, acc_bg, red_int, red_bg = intensities

    I_DD = grn_int - grn_bg
    I_DA = acc_int - acc_bg
    I_AA = red_int - red_bg

    if contains_nan(I_AA):
        F_DA = I_DA - (alpha * I_DD)
    else:
        F_DA = I_DA - (alpha * I_DD) - (delta * I_AA)

    return F_DA, I_DD, I_DA, I_AA


def calc_E(intensities, alpha=0, delta=0, clip_range=(-0.3, 1.3)):
    """
    Calculates raw FRET efficiency from donor (Dexc-Dem) and acceptor
    (Dexc-Aem). Note that iSMS has the option of subtracting background or not,
    and calculate E (apparent E) from that.
    """

    cmin, cmax = clip_range

    F_DA, I_DD, I_DA, I_AA = correct_DA(intensities, alpha, delta)

    E = F_DA / (I_DD + F_DA)
    E = np.clip(E, cmin, cmax, out=E)
    E = np.reshape(E, -1)

    return E


def calc_S(
    intensities, alpha=0, delta=0, beta=1, gamma=1, clip_range=(-0.3, 1.3)
):
    """
    Calculates raw calc_S from donor (Dexc-Dem), acceptor (Dexc-Aem) and direct
    emission of acceptor ("ALEX", Aexc-Aem) Note that iSMS has the option of
    subtracting background or not, and calculate S (apparent S) from that.
    """
    cmin, cmax = clip_range

    F_DA, I_DD, I_DA, I_AA = correct_DA(intensities, alpha, delta)

    inv_beta = 1 / beta

    S = (gamma * I_DD + F_DA) / (gamma * I_DD + F_DA + (inv_beta * I_AA))
    S = np.clip(S, cmin, cmax, out=S)
    S = np.reshape(S, -1)

    return S


def corrected_ES(
    intensities, alpha, delta, beta, gamma, clip_range=(-0.3, 1.3)
):
    """
    Calculates the fully corrected FRET and stoichiometry, given all the
    correction factors. This is only used for the combined 2D histogram,
    not for single traces.
    """
    cmin, cmax = clip_range

    F_DA, I_DD, I_DA, I_AA = correct_DA(intensities, alpha, delta)
    F_DD = gamma * I_DD
    F_AA = (1 / beta) * I_AA

    E = F_DA / (F_DA + F_DD)
    S = (F_DA + F_DD) / (F_DD + F_DA + F_AA)

    E = np.clip(E, cmin, cmax, out=E)
    S = np.clip(S, cmin, cmax, out=S)

    return E, S


def drop_bleached_frames(
    intensities, bleaches, max_frames=None, alpha=0, delta=0, beta=1, gamma=1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes all frames after bleaching
    """
    bleach = min_real(bleaches)

    if beta == 1 and gamma == 1:
        E_trace = calc_E(intensities, alpha, delta)
        S_trace = calc_S(intensities, alpha, delta)
    else:
        E_trace, S_trace = corrected_ES(intensities, alpha, delta, beta, gamma)

    E_trace_ = E_trace[:bleach][:max_frames]
    S_trace_ = S_trace[:bleach][:max_frames]

    return E_trace_, S_trace_


def alpha_factor(DD, DA):
    """
    Alpha factor for donor-only population.
    Use the donor and acceptor intensities minus background.
    """
    E_app = DA / (DD + DA)
    return np.median(E_app / (1 - E_app))


def delta_factor(DD, DA, AA):
    """
    Delta factor for acceptor-only population.
    Use the donor and acceptor intensities minus background.
    """
    S_app = (DD + DA) / (DD + DA + AA)
    return np.median(S_app / (1 - S_app))


def beta_gamma_factor(E_app, S_app):
    """
    Calculates global beta and gamma factors from apparent E and S (alpha and
    delta already applied).
    """
    # Sanitize inputs to avoid values tending towards infinity
    cond = (S_app > 0.3) & (S_app < 0.7)
    E_app = E_app[cond]
    S_app = S_app[cond]

    slope, intercept = leastsq_line(x=E_app, y=1 / S_app)

    beta = intercept + slope - 1
    gamma = (intercept - 1) / (intercept + slope - 1)

    if np.any(np.isnan((beta, gamma, slope, intercept))):
        beta, gamma = 1, 1

    return beta, gamma


def trim_ES(E: list, S: list):
    """
    Trims out-of-range values of E/S values
    """
    E, S = np.array(E), np.array(S)
    if contains_nan(S):
        # Use E only
        E = E[(E > -0.3) & (E < 1.3)]
    else:
        idx = (S > -0.3) & (S < 1.3)
        E, S = E[idx], S[idx]
    return E, S


def fit_gaussian_mixture(
    X: np.ndarray, min_n_components: int = 1, max_n_components: int = 1
):
    """
    Fits the best univariate gaussian mixture model, based on BIC
    If min_n_components == max_n_components, will lock to selected number, but
    still test all types of covariances
    """
    X = X.reshape(-1, 1)

    models, bic = [], []
    n_components_range = range(min_n_components, max_n_components + 1)
    cv_types = ["spherical", "tied", "diag", "full"]

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = sklearn.mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            models.append(gmm)
            bic.append(gmm.bic(X))

    best_gmm = models[int(np.argmin(bic))]
    print("number of components ", best_gmm.n_components)

    weights = best_gmm.weights_.ravel()
    means = best_gmm.means_.ravel()
    sigs = np.sqrt(best_gmm.covariances_.ravel())

    # Due to covariance type
    if len(sigs) != len(means):
        sigs = np.repeat(sigs, len(means))

    print("weights: ", weights)
    print("means: ", means)
    print("sigs: ", sigs)

    params = [(m, s, w) for m, s, w in zip(means, sigs, weights)]
    params = sorted(params, key=lambda tup: tup[0])

    return best_gmm, params


def fit_hmm(
    X: np.ndarray,
    fret: np.ndarray,
    lengths: List[int],
    covar_type: str,
    n_components: int,
):
    """
    Fits a Hidden Markov Model to traces. The traces are row-stacked, to provide
    a (t, c) matrix, where t is the total number of frames, and c is the
    channels
    """
    hmm_model = hmmlearn.hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covar_type,
        init_params="stmc",  # auto init all params
        algorithm="viterbi",
    )
    hmm_model.fit(X, lengths)

    states = hmm_model.predict(X, lengths)
    transmat = hmm_model.transmat_

    state_means, state_sigs = [], []
    for si in sorted(np.unique(states)):
        _, params = fit_gaussian_mixture(fret[states == si])
        for (m, s, _) in params:
            state_means.append(m)
            state_sigs.append(s)

    return states, transmat, state_means, state_sigs


def find_transitions(states, fret):
    """
    Finds transitions and their lifetimes, given states and FRET signal
    """
    hf = pd.DataFrame()
    hf["state"] = states
    hf["y_obs"] = fret
    hf["y_fit"] = hf.groupby(["state"], as_index=False)["y_obs"].transform(
        "median"
    )

    hf["time"] = hf["y_fit"].index + 1

    # Calculate lifetimes now, by making a copy to work on
    lf = hf.copy()

    # # Find y_after from y_before
    lf["state+1"] = np.roll(lf["state"], -1)

    # Find out when there's a change in state, depending on the minimum
    # transition size set
    lf["state_jump"] = lf["state"].transform(
        lambda group: (abs(group.diff()) > 0).cumsum()
    )

    # Drop duplicates
    lf.drop_duplicates(subset="state_jump", keep="last", inplace=True)

    # Find the difference for every time
    lf["lifetime"] = np.append(np.nan, np.diff(lf["time"]))

    lf = lf[["state", "state+1", "lifetime"]]
    lf = lf[:-1]
    lf.dropna(inplace=True)

    idealized = hf["y_fit"].values
    idealized_idx = hf["time"].values
    transitions = lf
    return idealized, idealized_idx, transitions


def sample_max_normalize_3d(X):
    """
    Sample-wise max-value normalization of 3D array (tensor).
    This is not feature-wise normalization, to keep the ratios between features
    intact!
    """
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    assert len(X.shape) == 3
    arr_max = np.max(X, axis=(1, 2), keepdims=True)
    X = X / arr_max
    return np.squeeze(X)


def seq_probabilities(yi, skip_threshold=0.5, skip_column=0):
    """
    Calculates class-wise probabilities over the entire trace for a one-hot
    encoded sequence prediction. Skips values where the first value is above
    threshold (bleaching).
    """
    assert len(yi.shape) == 2

    # Discard frames where bleaching (column 0) is above threshold (0.5)
    p = yi[yi[:, skip_column] < skip_threshold]
    if len(p) > 0:
        # Sum frame values for each class
        p = p.sum(axis=0) / len(p)

        # Normalize to 1
        p = p / p.sum()

        # don't ignore bleached frames entirely,
        # as it's easier to deal with a tiny number of edge cases
        # p[skip_column] = 0
    else:
        p = np.zeros(yi.shape[1])

    # sum static and dynamic smFRET scores (they shouldn't compete)
    confidence = p[4:].sum()
    return p, confidence


def find_bleach(p_bleach, threshold=0.5, window=7):
    """
    Finds bleaching given a list of frame-wise probabilities.
    The majority of datapoints in a given window must be above the threshold
    """
    is_bleached = scipy.signal.medfilt(p_bleach > threshold, window)
    bleach_frame = np.argmax(is_bleached)
    if bleach_frame == 0:
        bleach_frame = None
    return bleach_frame


def predict_single(xi, model):
    """
    Fixes dimensions to allow allow Keras prediction on a single sample

    np.newaxis adds an empty dimension to create a single-sample tensor, and
    np.squeeze removes the empty dimension after prediction
    """
    return np.squeeze(model.predict(xi[np.newaxis, :, :]))


def predict_batch(X, model, batch_size=256, progressbar: ProgressBar = None):
    """
    Predicts on batches in a loop
    """
    batches = (X.shape[0] // batch_size) + 1
    y_pred = []

    for i in range(batches):
        i1 = i * batch_size
        i2 = i1 + batch_size
        y_pred.append(model.predict_on_batch(X[i1:i2]))

        if progressbar is not None:
            progressbar.increment()
            if progressbar.wasCanceled():
                break

    return np.row_stack(y_pred)


def count_adjacent_values(arr):
    """
    Returns start index and length of segments of equal values.

    Example for plotting several axvspans:
    --------------------------------------
    adjs, lns = lib.count_adjacent_true(score)
    t = np.arange(1, len(score) + 1)

    for ax in axes:
        for starts, ln in zip(adjs, lns):
            alpha = (1 - np.mean(score[starts:starts + ln])) * 0.15
            ax.axvspan(xmin = t[starts], xmax = t[starts] + (ln - 1),
            alpha = alpha, color = "red", zorder = -1)
    """
    arr = arr.ravel()

    n = 0
    same = [(g, len(list(l))) for g, l in itertools.groupby(arr)]
    starts = []
    lengths = []
    for v, l in same:
        _len = len(arr[n : n + l])
        _idx = n
        n += l
        lengths.append(_len)
        starts.append(_idx)
    return starts, lengths


def all_equal(iterator):
    """
    Checks if all elements in an iterator are equal
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def min_real(ls) -> Union[Union[int, float], None]:
    """
    Finds the minimum value of a list, ignoring None.
    Returns None if all values are None
    """
    ls = np.array(ls)
    ls = ls[ls != None]
    return None if len(ls) == 0 else min(ls)


def contour_2d(
    xdata,
    ydata,
    bandwidth=0.1,
    n_colors=2,
    kernel="gaussian",
    extend_grid=1,
    shade_lowest=False,
    resolution=100,
    cbins="auto",
):
    """
    Calculates the 2D kernel density estimate for a dataset.

    Valid kernels for sklearn are
    'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'

    Example
    -------
    x, y, z, levels = countour_2d(x, y, shade_lowest = False)

    fig, ax = plt.subplots()
    c = ax.contourf(x,y,z, levels=levels, cmap = "inferno")

    Alternatively, unpack like

    contour = countour_2d(xdata, ydata)
    c = ax.contourf(*contour)

    For optional colorbar, add fig.colorbar(c)
    """

    if kernel == "epa":
        kernel = "epanechnikov"

    if bandwidth == "auto":
        bandwidth = (len(xdata) * 4 / 4.0) ** (-1.0 / 6)

    # Stretch the min/max values to make sure that the KDE goes beyond the
    # outermost points
    meanx = np.mean(xdata) * extend_grid
    meany = np.mean(ydata) * extend_grid

    # Create a grid for KDE
    x, y = np.mgrid[
        min(xdata) - meanx : max(xdata) + meanx : complex(resolution),
        min(ydata) - meany : max(ydata) + meany : complex(resolution),
    ]

    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([xdata, ydata])

    # Define KDE with specified bandwidth
    kernel_sk = sklearn.neighbors.KernelDensity(
        kernel=kernel, bandwidth=bandwidth
    ).fit(list(zip(*values)))
    z = np.exp(kernel_sk.score_samples(list(zip(*positions))))

    z = np.reshape(z.T, x.shape)

    if not shade_lowest:
        n_colors += 1

    locator = matplotlib.ticker.MaxNLocator(n_colors, min_n_ticks=n_colors)

    if type(cbins) == np.ndarray:
        levels = cbins
    elif cbins is "auto":
        levels = locator.tick_values(z.min(), z.max())
    else:
        raise ValueError

    if not shade_lowest:
        levels = levels[1:]

    return x, y, z, levels


def estimate_bw(n, d, factor):
    """
    Estimate optimal bandwidth parameter, based on Silverman's rule
    (see SciPy docs)

    Parameters
    ----------
    n:
        Number of data points
    d:
        Number of dimensions
    factor:
        Multiply by constant for better adjustment

    Returns
    -------
    Optimal smoothing bandwidth
    """
    return ((n * (d + 2) / 4.0) ** (-1.0 / (d + 4))) * factor ** 2


def histpoints_w_err(
    data, bins, density, remove_empty_bins=False, least_count=1
):
    """
    Converts unbinned data to x,y-curvefitable points with Poisson errors.

    Parameters
    ----------
    data:
        Unbinned input data
    bins:
        Number of bins, or defined bins
    density:
        Whether to normalize histogram (use normalization factor for plots)
    remove_empty_bins:
        Whether to remove bins with less than a certain number of counts,
        to assume roughly gaussian errors on points (default 5)
    least_count:
        See above. Default is 5, according to theory

    Returns
    -------
    x, y, y-error points and normalization constant

    """
    counts, bin_edges = np.histogram(data, bins=bins, density=density)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_err = np.sqrt(counts)

    # Get the normalization constant
    unnorm_counts, bin_edges = np.histogram(data, bins=bins, density=False)

    # Generate fitting points
    if remove_empty_bins:
        # regardless of normalization, get actual counts per bin
        true_counts, _ = np.histogram(data, bins, density=False)
        # filter along counts, to remove any value in the same position as
        # an empty bin
        x = bin_centers[true_counts >= int(least_count)]
        y = counts[true_counts >= int(least_count)]
        sy = bin_err[true_counts >= int(least_count)]
    else:
        x, y, sy = bin_centers, counts, bin_err

    norm_const = np.sum(unnorm_counts * (bin_edges[1] - bin_edges[0]))
    if density:
        sy = 1 / norm_const * sy
    return x, y, sy, norm_const


def estimate_binwidth(x):
    """Estimate optimal binwidth by the Freedman-Diaconis rule."""
    return 2 * scipy.stats.iqr(x) / np.size(x) ** (1 / 3)


def exp_function(x, N, lam):
    e = np.exp
    return N * (lam * e(-lam * x))
