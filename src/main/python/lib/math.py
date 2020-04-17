import multiprocessing

multiprocessing.freeze_support()

import matplotlib
import matplotlib.ticker
import sklearn.neighbors
import lib.misc
from ui.misc import ProgressBar
from typing import Union, Tuple, List
import scipy.signal
import scipy.signal
import scipy.optimize
import scipy.stats
import sklearn.cluster
import sklearn.mixture
import hmmlearn.hmm
import scipy.stats
import pandas as pd
import numpy as np
import pomegranate as pg
from retrying import retry, RetryError
from tqdm import tqdm
from lib.misc import timeit

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
        E = E[(E > -0.3) & (E < 1.3)]  # original line
    else:
        idx = (S > -0.3) & (S < 1.3)
        E, S = E[idx], S[idx]
    return E, S


def fit_gaussian_mixture(
    X: np.ndarray,
    min_n_components: int = 1,
    max_n_components: int = 1,
    strict_bic: [bool, int, float] = False,
    verbose: bool = False,
):
    """
    Fits the best univariate gaussian mixture model, based on BIC
    If min_n_components == max_n_components, will lock to selected number, but
    still test all types of covariances
    """
    X = X.reshape(-1, 1)

    models, bic = [], []
    n_components_range = range(
        min(min_n_components, max_n_components),
        max(min_n_components, max_n_components) + 1,
    )
    cv_types = ["spherical", "tied", "diag", "full"]

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = sklearn.mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type,
            )
            gmm.fit(X)
            models.append(gmm)
            if strict_bic is not False:
                strictness = strict_bic if strict_bic is not bool else 2
                b = -2 * gmm.score(X) * X.shape[0] + (
                    gmm._n_parameters() ** strictness
                ) * np.log(2 * X.shape[0])
            else:
                b = gmm.bic(X)
            bic.append(b)

    best_gmm = models[int(np.argmin(bic))]

    weights = best_gmm.weights_.ravel()
    means = best_gmm.means_.ravel()
    sigs = np.sqrt(best_gmm.covariances_.ravel())

    # Due to covariance type
    if len(sigs) != len(means):
        sigs = np.repeat(sigs, len(means))
    if verbose:
        print("number of components ", best_gmm.n_components)
        print("weights: ", weights)
        print("means: ", means)
        print("sigs: ", sigs)

    params = [(m, s, w) for m, s, w in zip(means, sigs, weights)]
    params = sorted(params, key=lambda tup: tup[0])

    return best_gmm, params


def fit_hmm_pg(
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
    model = pg.HiddenMarkovModel.from_samples(
        pg.NormalDistribution,
        name=None,
        n_components=n_components,
        X=X,
        n_jobs=-1,
        # callbacks=[pgc.ModelCheckpoint(name=name)],
    )

    hmm_model = hmmlearn.hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        min_covar=100,
        init_params="stmc",  # auto init all params
        algorithm="viterbi",
    )
    hmm_model.fit(X, lengths)
    print("covariances are: ", hmm_model.covars_)

    states = hmm_model.predict(X, lengths)
    transmat = hmm_model.transmat_

    state_means, state_sigs = [], []
    for si in sorted(np.unique(states)):
        _, params = fit_gaussian_mixture(fret[states == si])
        for (m, s, _) in params:
            state_means.append(m)
            state_sigs.append(s)

    return states, transmat, state_means, state_sigs


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
    X = X - np.mean(X)
    X = X / np.std(X)

    hmm_model = hmmlearn.hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        min_covar=100,
        init_params="stmc",  # auto init all params
        algorithm="viterbi",
    )
    hmm_model.fit(X, lengths)
    print("covariances are: ", hmm_model.covars_)

    states = hmm_model.predict(X, lengths)
    transmat = hmm_model.transmat_

    state_means, state_sigs = [], []
    for si in sorted(np.unique(states)):
        _, params = fit_gaussian_mixture(fret[states == si])
        for (m, s, _) in params:
            state_means.append(m)
            state_sigs.append(s)

    return states, transmat, state_means, state_sigs


def get_hmm_model(X, n_components=5, name=None):
    return pg.HiddenMarkovModel.from_samples(
            pg.NormalDistribution,
            name=name,
            n_components=n_components,
            X=X,
            n_jobs=-1,
            # callbacks=[pgc.ModelCheckpoint(name=name)],
        )


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
        print(X[i1:i2].shape)

        y_pred.append(model.predict_on_batch(X[i1:i2]))

        if progressbar is not None:
            progressbar.increment()
            if progressbar.wasCanceled():
                break

    return np.row_stack(y_pred)


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
    diagonal: bool = False,
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

    if kernel.startswith("epa"):
        kernel = "epanechnikov"

    if bandwidth == "auto":
        bandwidth = (len(xdata) * 4 / 4.0) ** (-1.0 / 6)

    # Stretch the min/max values to make sure that the KDE goes beyond the
    # outermost points
    meanx = np.mean(xdata) * extend_grid
    meany = np.mean(ydata) * extend_grid

    # Create a grid for KDE
    if diagonal:
        vmin = min(min(xdata), min(ydata))
        vmax = max(max(xdata), max(ydata))

        mean = np.mean(np.concatenate([xdata, ydata])) * extend_grid

        x, y = np.mgrid[
            vmin - mean : vmax + mean : complex(resolution),
            vmin - mean : vmax + mean : complex(resolution),
        ]
    else:
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


def generate_traces(
    n_traces,
    state_means="random",
    random_k_states_max=5,
    min_state_diff=0.1,
    D_lifetime=400,
    A_lifetime=200,
    blink_prob=0.05,
    bleed_through=0,
    aa_mismatch=(-0.3, 0.3),
    trace_length=200,
    trans_prob=0.1,
    noise=0.08,
    trans_mat=None,
    au_scaling_factor=1,
    aggregation_prob=0.1,
    max_aggregate_size=100,
    null_fret_value=-1,
    acceptable_noise=0.25,
    scramble_prob=0.3,
    gamma_noise_prob=0.5,
    merge_labels=False,
    discard_unbleached=False,
    progressbar_callback=None,
    callback_every=1,
):
    """
    Parameters
    ----------
    n_traces:
        Number of traces to generate
    state_means:
        Mean FRET value. Add multiple values for multiple states
    random_k_states_max:
        If state_means = "random", randomly selects at most k FRET states
    min_state_diff:
        If state_means = "random", randomly spaces states with a minimum
        distance
    D_lifetime:
        Lifetime of donor fluorophore, as drawn from exponential distribution.
        Set to None if fluorophore shouldn't bleach.
    A_lifetime:
        Lifetime of acceptor fluorophore, as drawn from exponential
        distribution. Set to None if fluorophore shouldn't bleach.
    blink_prob:
        Probability of observing photoblinking in a trace.
    bleed_through:
        Donor bleed-through into acceptor channel, as a fraction of the signal.
    aa_mismatch:
        Acceptor-only intensity mis-correspondence, as compared to DD+DA signal.
        Set as a value or range. A value e.g. 0.1 corresponds to 110% of the
        DD+DA signal. A range (-0.3, 0.3) corresponds to 70% to 130% of DD+DA
        signal.
    trace_length:
        Simulated recording length of traces. All traces will adhere to this
        length.
    trans_prob:
        Probability of transitioning from one state to another, given the
        transition probability matrix. This can also be overruled by a supplied
        transition matrix (see trans_mat parameter).
    noise:
        Noise added to a trace, as generated from a Normal(0, sigma)
        distribution. Sigma can be either a value or range.
    trans_mat:
        Transition matrix to be provided instead of the quick trans_prob
        parameter.
    au_scaling_factor:
        Arbitrary unit scaling factor after trace generation. Can be value or
        range.
    aggregation_prob:
        Probability of trace being an aggregate. Note that this locks the
        labelled molecule in a random, fixed FRET state.
    max_aggregate_size:
        Maximum number of labelled molecules in an aggregate.
    null_fret_value:
        Whether to set a specific value for the no-longer-viable *ground truth*
        FRET, e.g. -1, to easily locate it for downstream processing.
    acceptable_noise:
        Maximum acceptable noise level before trace is labelled as "noisy". If
        acceptable_noise is above the upper range of noise, no "noisy" traces
        will be generated.
    scramble_prob:
        Probability that the trace will end up being scrambled. This stacks with
        aggregation.
    gamma_noise_prob:
        Probability to multiply centered Gamma(1, 0.11) to each frame's noise,
        to make the data appear less synthetic
    merge_labels:
        Merges (dynamic, static) and (aggregate, noisy, scrambled) to deal with
        binary labels only
    discard_unbleached:
        Whether to discard traces that don't fully bleach to background.
    callback_every:
        How often to callback to the progressbar
    progressbar_callback:
        Progressbar callback object
    """

    def _E(DD, DA):
        return DA / (DD + DA)

    def _S(DD, DA, AA):
        return (DD + DA) / (DD + DA + AA)

    def _DD(E):
        return 1 - E

    def _DA(DD, E):
        return -(DD * E) / (E - 1)

    def _AA(E):
        return np.ones(len(E))

    @retry
    def generate_state_means(min_diff, k_states):
        """Returns random values and retries if they are too closely spaced"""
        states = np.random.uniform(0.01, 0.99, k_states)
        diffs = np.diff(sorted(states))
        if any(diffs < min_diff):
            raise RetryError
        return states

    def generate_fret_states(kind, state_means, trans_mat, trans_prob):
        """Creates artificial FRET states"""
        if all(isinstance(s, float) for s in state_means):
            kind = "defined"

        rand_k_states = np.random.randint(1, random_k_states_max + 1)

        if kind == "aggregate":
            state_means = np.random.uniform(0, 1)
            k_states = 1
        elif kind == "random":
            k_states = rand_k_states
            state_means = generate_state_means(min_state_diff, k_states)
        else:
            if np.size(state_means) <= random_k_states_max:
                # Pick the same amount of k states as state means given
                k_states = np.size(state_means)
            else:
                # Pick no more than k_states_max from the state means (e.g.
                # given [0.1, 0.2, 0.3, 0.4, 0.5] use only
                # random_k_states_max of these)
                k_states = rand_k_states
                state_means = np.random.choice(
                    state_means, size=k_states, replace=False
                )

        if type(state_means) == float:
            dists = [pg.NormalDistribution(state_means, 0)]
        else:
            dists = [pg.NormalDistribution(m, 1e-16) for m in state_means]

        starts = np.random.uniform(0, 1, size=k_states)
        starts /= starts.sum()

        # Generate arbitrary transition matrix
        if trans_mat is None:
            trans_mat = np.empty([k_states, k_states])
            trans_mat.fill(trans_prob)
            np.fill_diagonal(trans_mat, 1 - trans_prob)

            # Make sure that each row/column sums to exactly 1
            if trans_prob != 0:
                stay_prob = 1 - trans_prob
                remaining_prob = 1 - trans_mat.sum(axis=0)
                trans_mat[trans_mat == stay_prob] += remaining_prob

        # Generate HMM model
        model = pg.HiddenMarkovModel.from_matrix(
            trans_mat, distributions=dists, starts=starts
        )
        model.bake()

        E_true = np.array(model.sample(n=1, length=trace_length))
        E_true = np.squeeze(E_true).round(4)
        return E_true

    def scramble(DD, DA, AA, cls, label):
        """Scramble trace for model robustness"""

        modify_trace = np.random.choice(("DD", "DA", "AA"))
        if modify_trace == "AA":
            c = AA
        elif modify_trace == "DA":
            c = DA
        elif modify_trace == "DD":
            c = DD
        else:
            raise ValueError

        c[c != 0] = 1
        # Create a sign wave and merge with trace
        sinwave = np.sin(np.linspace(-10, np.random.randint(0, 1), len(DD)))
        sinwave[c == 0] = 0
        sinwave = sinwave ** np.random.randint(5, 10)
        c += sinwave * 0.4
        # Fix negatives
        c = np.abs(c)

        # Correlate heavily
        DA *= AA * np.random.uniform(0.7, 1)
        AA *= DA * np.random.uniform(0.7, 1)
        DD *= AA * np.random.uniform(0.7, 1)

        # Add dark state
        add_dark = np.random.choice(("add", "noadd"))
        if add_dark == "add":
            dark_state_start = np.random.randint(0, 40)
            dark_state_time = np.random.randint(10, 40)
            dark_state_end = dark_state_start + dark_state_time
            DD[dark_state_start:dark_state_end] = 0

        # Add noise
        if np.random.uniform(0, 1) < 0.1:
            noise_start = np.random.randint(1, trace_length)
            noise_time = np.random.randint(10, 50)
            noise_end = noise_start + noise_time
            if noise_end > trace_length:
                noise_end = trace_length

            DD[noise_start:noise_end] *= np.random.normal(
                1, 1, noise_end - noise_start
            )

        # Flip traces
        flip_trace = np.random.choice(("flipDD", "flipDA", "flipAA"))
        if flip_trace == "flipDD":
            DD = np.flip(DD)
        elif flip_trace == "flipAA":
            AA = np.flip(AA)
        elif flip_trace == "flipDA":
            DA = np.flip(DA)

        DD, DA, AA = [np.abs(x) for x in (DD, DA, AA)]

        label.fill(cls["scramble"])
        return DD, DA, AA, label

    def generate_single_trace(*args):
        """Function to generate a single trace"""
        (
            i,
            trans_prob,
            au_scaling_factor,
            noise,
            bleed_through,
            aa_mismatch,
            scramble_prob,
        ) = [np.array(arg) for arg in args]

        # Simple table to keep track of labels
        cls = {
            "bleached": 0,
            "aggregate": 1,
            "noisy": 2,
            "scramble": 3,
            "1-state": 4,
            "2-state": 5,
            "3-state": 6,
            "4-state": 7,
            "5-state": 8,
        }

        name = [i.tolist()] * trace_length
        frames = np.arange(1, trace_length + 1, 1)

        if np.random.uniform(0, 1) < aggregation_prob:
            is_aggregated = True
            E_true = generate_fret_states(
                kind="aggregate",
                trans_mat=trans_mat,
                trans_prob=0,
                state_means=state_means,
            )
            if max_aggregate_size >= 2:
                aggregate_size = np.random.randint(2, max_aggregate_size + 1)
            else:
                raise ValueError("Can't have an aggregate of size less than 2")
            np.random.seed()
            n_pairs = np.random.poisson(aggregate_size)
            if n_pairs == 0:
                n_pairs = 2
        else:
            is_aggregated = False
            n_pairs = 1
            trans_prob = np.random.uniform(trans_prob.min(), trans_prob.max())
            E_true = generate_fret_states(
                kind=state_means,
                trans_mat=trans_mat,
                trans_prob=trans_prob,
                state_means=state_means,
            )

        DD_total, DA_total, AA_total = [], [], []
        first_bleach_all = []

        for j in range(n_pairs):
            np.random.seed()
            if D_lifetime is not None:
                bleach_D = int(np.ceil(np.random.exponential(D_lifetime)))
            else:
                bleach_D = None

            if A_lifetime is not None:
                bleach_A = int(np.ceil(np.random.exponential(A_lifetime)))
            else:
                bleach_A = None

            first_bleach = lib.misc.min_none((bleach_D, bleach_A))
            first_bleach_all.append(first_bleach)

            # Calculate from underlying E
            DD = _DD(E_true)
            DA = _DA(DD, E_true)
            AA = _AA(E_true)

            # In case AA intensity doesn't correspond exactly to donor
            # experimentally (S will be off)
            AA += np.random.uniform(aa_mismatch.min(), aa_mismatch.max())

            # If donor bleaches first
            if first_bleach is not None:
                if first_bleach == bleach_D:
                    # Donor bleaches
                    DD[bleach_D:] = 0
                    # DA goes to zero because no energy is transferred
                    DA[bleach_D:] = 0

                # If acceptor bleaches first
                elif first_bleach == bleach_A:
                    # Donor is 1 when there's no acceptor
                    DD[bleach_A:bleach_D] = 1
                    if is_aggregated and n_pairs <= 2:
                        # Sudden spike for small aggregates to mimic
                        # observations
                        spike_len = np.min((np.random.randint(2, 10), bleach_D))
                        DD[bleach_A : bleach_A + spike_len] = 2

            # No matter what, zero each signal after its own bleaching
            if bleach_D is not None:
                DD[bleach_D:] = 0
            if bleach_A is not None:
                DA[bleach_A:] = 0
                AA[bleach_A:] = 0

            # Append to total fluorophore intensity per channel
            DD_total.append(DD)
            DA_total.append(DA)
            AA_total.append(AA)

        DD, DA, AA = [np.sum(x, axis=0) for x in (DD_total, DA_total, AA_total)]

        # Initialize -1 label for whole trace
        label = np.zeros(trace_length)
        label.fill(-1)

        # Calculate when a channel is bleached. For aggregates, it's when a
        # fluorophore channel hits 0 from bleaching (because 100% FRET not
        # considered possible)
        if is_aggregated:
            # First bleaching for
            bleach_DD_all = np.argmax(DD == 0)
            bleach_DA_all = np.argmax(DA == 0)
            bleach_AA_all = np.argmax(AA == 0)

            # Find first bleaching overall
            first_bleach_all = lib.misc.min_none(
                (bleach_DD_all, bleach_DA_all, bleach_AA_all)
            )
            if first_bleach_all == 0:
                first_bleach_all = None
            label.fill(cls["aggregate"])
        else:
            # Else simply check whether DD or DA bleaches first from lifetimes
            first_bleach_all = lib.misc.min_none(first_bleach_all)

        # Save unblinked fluorophores to calculate E_true
        DD_no_blink, DA_no_blink = DD.copy(), DA.copy()

        # No blinking in aggregates (excessive/complicated)
        if not is_aggregated and np.random.uniform(0, 1) < blink_prob:
            blink_start = np.random.randint(1, trace_length)
            blink_time = np.random.randint(1, 15)

            # Blink either donor or acceptor
            if np.random.uniform(0, 1) < 0.5:
                DD[blink_start : (blink_start + blink_time)] = 0
                DA[blink_start : (blink_start + blink_time)] = 0
            else:
                DA[blink_start : (blink_start + blink_time)] = 0
                AA[blink_start : (blink_start + blink_time)] = 0

        if first_bleach_all is not None:
            label[first_bleach_all:] = cls["bleached"]
            E_true[first_bleach_all:] = null_fret_value

        for x in (DD, DA, AA):
            # Bleached points get label 0
            label[x == 0] = cls["bleached"]

        if is_aggregated:
            first_bleach_all = np.argmin(label)
            if first_bleach_all == 0:
                first_bleach_all = None

        # Scramble trace, but only if contains 1 or 2 pairs (diminishing
        # effect otherwise)
        is_scrambled = False
        if np.random.uniform(0, 1) < scramble_prob and n_pairs <= 2:
            DD, DA, AA, label = scramble(
                DD=DD, DA=DA, AA=AA, cls=cls, label=label
            )
            is_scrambled = True

        # Figure out bleached places before true signal is modified:
        is_bleached = np.zeros(trace_length)
        for x in (DD, DA, AA):
            is_bleached[x == 0] = 1

        # Add donor bleed-through
        DD_bleed = np.random.uniform(bleed_through.min(), bleed_through.max())
        DA[DD != 0] += DD_bleed

        # Re-adjust E_true to match offset caused by correction factors
        # so technically it's not the true, corrected FRET, but actually the
        # un-noised
        E_true[E_true != null_fret_value] = _E(
            DD_no_blink[E_true != null_fret_value],
            DA_no_blink[E_true != null_fret_value],
        )

        # Add gaussian noise
        noise = np.random.uniform(noise.min(), noise.max())
        x = [s + np.random.normal(0, noise, len(s)) for s in (DD, DA, AA)]

        # Add centered gamma noise
        if np.random.uniform(0, 1) < gamma_noise_prob:
            for signal in x:
                gnoise = np.random.gamma(1, noise * 1.1, len(signal))
                signal += gnoise
                signal -= np.mean(gnoise)

        # Scale trace to AU units and calculate observed E and S as one would
        # in real experiments
        au_scaling_factor = np.random.uniform(
            au_scaling_factor.min(), au_scaling_factor.max()
        )
        DD, DA, AA = [s * au_scaling_factor for s in x]

        E_obs = _E(DD, DA)
        S_obs = _S(DD, DA, AA)

        # FRET from fluorophores that aren't bleached
        E_unbleached = E_obs[:first_bleach_all]
        E_unbleached_true = E_true[:first_bleach_all]

        # Count actually observed states, because a slow system might not
        # transition in the observation window
        observed_states = np.unique(E_true[E_true != null_fret_value])

        # Calculate noise level for each FRET state, and check if it
        # surpasses the limit
        is_noisy = False
        for state in observed_states:
            noise_level = np.std(E_unbleached[E_unbleached_true == state])
            if noise_level > acceptable_noise:
                label[label != cls["bleached"]] = cls["noisy"]
                is_noisy = True

        # For all FRET traces, assign the number of states observed
        if not any((is_noisy, is_aggregated, is_scrambled)):
            for i in range(5):
                k_states = i + 1
                if len(observed_states) == k_states:
                    label[label != cls["bleached"]] = cls[
                        "{}-state".format(k_states)
                    ]

        # Bad traces don't contain FRET
        if any((is_noisy, is_aggregated, is_scrambled)):
            E_true.fill(-1)

        # Everything that isn't FRET is 0, and FRET is 1
        if merge_labels:
            label[label <= 3] = 0
            label[label >= 4] = 1

        if discard_unbleached and label[-1] != cls["bleached"]:
            return pd.DataFrame()

        # Calculate difference between states if >=2 states and actual smFRET
        if label[0] in [5, 6, 7, 8]:
            min_diff = np.min(np.diff(np.unique(E_unbleached_true)))
        else:
            min_diff = np.nan

        bg = np.zeros_like(DD)

        # Columns pre-fixed with underscore contain metadata, and only the
        # first value should be used (repeated because table structure)
        trace = pd.DataFrame(
            {
                "D-Dexc-rw": DD,
                "A-Dexc-rw": DA,
                "A-Aexc-rw": AA,
                "D-Dexc-bg": bg,
                "A-Dexc-bg": bg,
                "A-Aexc-bg": bg,
                "E": E_obs,
                "E_true": E_true,
                "S": S_obs,
                "frame": frames,
                "name": name,
                "label": label,
                "_bleaches_at": np.array(first_bleach_all).repeat(trace_length),
                "_noise_level": np.array(noise).repeat(trace_length),
                "_min_state_diff": np.array(min_diff).repeat(trace_length),
            }
        )
        trace.replace([np.inf, -np.inf], np.nan, inplace=True)
        trace.fillna(method="pad", inplace=True)
        return trace

    processes = range(n_traces)
    traces = []
    for i in processes:
        traces.append(
            generate_single_trace(
                i,
                trans_prob,
                au_scaling_factor,
                noise,
                bleed_through,
                aa_mismatch,
                scramble_prob,
            )
        )
        if progressbar_callback is not None and (i % callback_every) == 0:
            progressbar_callback.increment()

    traces = pd.concat(traces) if len(traces) > 1 else traces[0]
    return traces


def func_double_exp(
    _x: np.ndarray, _lambda_1: float, _lambda_2: float, _k: float
):
    if _k > 1.0:
        raise ValueError(f"_k of value {_k:.2f} is larger than 1!")
    if _k < 0:
        raise ValueError(f"_k of value {_k:.2f} is smaller than 1!")
    _exp1 = _lambda_1 * np.exp(-1 * _lambda_1 * _x)
    _exp2 = _lambda_2 * np.exp(-1 * _lambda_2 * _x)

    return _k * _exp1 + (1 - _k) * _exp2


def func_exp(
    _x: np.ndarray, _lambda,
):
    return _lambda * np.exp(-_lambda * _x)


def loglik_single(x: np.ndarray, _lambda):
    """
    Returns Negative Loglikelihood for a single exponential with given param and given observations
    """
    return -1 * np.sum(np.log(func_exp(x, _lambda)))


def loglik_double(x: np.ndarray, _lambda_1: float, _lambda_2: float, _k: float):
    """
    Returns Negative Loglikelihood for a double exponential with given params and given observations
    """
    return -1 * np.sum(np.log(func_double_exp(x, _lambda_1, _lambda_2, _k)))


def fit_and_compare_exp_funcs(
    arr,
    x0: Union[None, Tuple[float]] = (2.0, 3.0, 0.55),
    verbose=False,
    meth="l-bfgs-b",
):
    ftol = 2.220446049250313e-09
    arr = np.nan_to_num(arr,)
    if x0 is None:
        fit_loc, fit_scale = scipy.stats.expon.fit(arr, floc=0)
        fit_lambda = 1.0 / fit_scale
        x0 = (fit_lambda, fit_lambda, 0.55)

    def lh_single(l):
        return loglik_single(arr, l)

    def lh_double(x):
        return loglik_double(arr, *x)

    res1 = scipy.optimize.minimize(
        lh_single, x0=np.array(x0[0]), method=meth, bounds=[(0.0, None)]
    )
    errs1 = np.zeros(len(res1.x))
    tmp_i = np.zeros(len(res1.x))
    for i in range(len(res1.x)):
        tmp_i[i] = 1.0
        hess_inv_i = res1.hess_inv(tmp_i)[i]
        uncertainty_i = np.sqrt(max(1, abs(res1.fun)) * ftol * hess_inv_i)
        tmp_i[i] = 0.0
        errs1[i] = uncertainty_i

    llh_1 = -res1.fun
    bic_1 = np.log(len(arr)) * 1 - 2 * llh_1
    if verbose:
        print("Params for single exp:")
        print(res1.x)
        print("Uncertainties: ")
        print(errs1)
        print(f"BIC : {bic_1:.6f}")

    res2 = scipy.optimize.minimize(
        lh_double,
        x0=np.array(x0),
        method=meth,
        bounds=[(0.0, None), (0.0, None), (0.01, 0.99)],
    )

    errs2 = np.zeros(len(res2.x))
    tmp_i = np.zeros(len(res2.x))
    for i in range(len(res2.x)):
        tmp_i[i] = 1.0
        hess_inv_i = res2.hess_inv(tmp_i)[i]
        uncertainty_i = np.sqrt(max(1, abs(res2.fun)) * ftol * hess_inv_i)
        tmp_i[i] = 0.0
        errs2[i] = uncertainty_i

    llh_2 = -res2.fun
    bic_2 = np.log(len(arr)) * 3 - 2 * llh_2
    if verbose:
        print("Params for double exp:")
        print(res2.x)
        print("Uncertainties: ")
        print(errs2)
        print(f"BIC : {bic_2:.6f}")

    return {
        'BEST': 'SINGLE' if bic_1 < bic_2 else 'DOUBLE',
        'SINGLE_LLH': llh_1,
        'SINGLE_BIC': bic_1,
        'SINGLE_PARAM': res1.x,
        'SINGLE_ERRS': errs1,
        'DOUBLE_LLH': llh_2,
        'DOUBLE_BIC': bic_2,
        'DOUBLE_PARAM': res2.x,
        'DOUBLE_ERRS': errs2,
    }


def corrcoef_lags(x, y, n_lags: int = 5):
    if not isinstance(n_lags, int):
        n_lags = int(n_lags)
    if n_lags > len(x):
        n_lags = len(x) - 1
    cs = np.zeros(2 * n_lags + 1)
    for i in range(2 * n_lags + 1):
        if i <= n_lags:  # 0,1,2,3,4
            _x = np.pad(x, (0, n_lags), mode="constant")
            _y = np.pad(y, (n_lags - i, i), mode="constant")  # "edge" flag?
        else:  # 5,6,7,8,9
            _x = np.pad(x, (i - n_lags, 2 * n_lags - i), mode="constant")
            _y = np.pad(y, (0, n_lags), mode="constant")
        cs[i], _ = scipy.stats.pearsonr(_x, _y)
    return cs


def correct_corrs(corrs):
    maxlen = max(len(arr) for arr in corrs)
    _corrs = np.zeros((len(corrs), maxlen))
    for j, corr in enumerate(corrs):
        _l = len(corr)
        if _l == maxlen:
            _corrs[j] = corr
        else:
            missing_per_side = int(int(maxlen - _l) / 2)
            _corrs[j] = np.pad(
                corr,
                (missing_per_side, missing_per_side),
                "constant",
                constant_values=(np.nan, np.nan),
            )
    return _corrs
