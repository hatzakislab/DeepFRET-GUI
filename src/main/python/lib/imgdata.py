import multiprocessing
multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
from scipy.interpolate import UnivariateSpline
import skimage.exposure
import skimage.feature
import skimage.filters
import numpy as np
import itertools
import pandas as pd


def image_quadrants(height, width):
    """
    Returns indices for quadrants of image, given height and width.
    For example, img[top, left] will return the top left quadrant of the image
    """

    if height != width:
        raise ValueError("Expected image to be square but got {}x{}".format(height, width))

    m_yi = height // 2
    m_xi = width // 2

    top = slice(0, m_yi)
    bottom = slice(m_yi, height)
    left = slice(0, m_xi)
    right = slice(m_xi, width)

    return top, bottom, left, right


def image_channels(cmax):
    """
    Given a currentMovie where images from different channels occur in the sequence
    frame1: c1_1, c2_1, c3_1, c4_1
    frame2: c1_2, c2_2, c3_2, c4_2
    frame3: c1_3, c2_3, c3_3, c4_3

    cmax is 4, so the 4 channels can be extracted with the returned slices
    [0::4], [1::4], [2::4], [3::4]

    Example:
    c1, c2, c3, c4 = image_channels(cmax = 4)
    """

    channels = []
    for c in range(cmax):
        channels.append(slice(c, None, cmax))

    return channels


def subtract_background(arr, deg=2, s=1e4, by= "row", return_bg_only=False, **filter_kwargs):
    """
    Subtracts background with a row-wise spline fit or filter, to correct for non-uniform illumination profile

    Parameters
    ----------
    arr:
        Input image array
    deg:
        Spline polynomial degree
    s:
        Spline smoothing factor
    by:
        Row-wise or column-wise spline fitting. Default is row, because this also eliminates column noise.
    return_bg_only:
        Whether to return the fitted background only

    Returns
    -------
    Fitted background of array. Should be subtracted from the image if bg_only is True.
    """

    if len(arr.shape) == 2:
        rows, columns = arr.shape
    elif len(arr.shape) == 3:
        time, rows, columns = arr.shape
    else:
        raise ValueError("Either too few or too many dimensions")

    _ = []
    if by == "column":
        # Columnwise spline
        ix = np.arange(0, rows)  # each column iterated over has the length of the number of rows
        ls = [UnivariateSpline(ix, arr[:, i], k=deg, s=s)(ix) for i in range(columns)]
        columnwise = np.column_stack(ls)
        _.append(columnwise)

    if by == "row":
        # Rowwise spline
        ix = np.arange(0, columns)  # each row iterated over has the length of the number of columns
        ls = [UnivariateSpline(ix, arr[i, :], k=deg, s=s)(ix) for i in range(rows)]
        rowwise = np.row_stack(ls)
        _.append(rowwise)

    if by == "filter":  # This is faster
        ls = skimage.filters.gaussian(arr, **filter_kwargs)
        _.append(ls)

    bg, = _

    return bg if return_bg_only else arr - bg


def zero_one_scale(arr: np.ndarray):
    """
    Scales all values to the range [0, 1].

    Parameters
    ----------
    arr:
        numpy.ndarray

    Returns
    -------
    Normalized array
    """
    normalized = (arr - arr.min()) / (arr.max() - arr.min())
    return normalized


def rescale_intensity(image: np.ndarray, range):
    """
    Rescales image intensity by clipping the range, followed by normalization.

    Parameters
    ----------
    image:
        Image to rescale (or other numpy.ndarray)
    range:
        Clipping range. Should be between 0 and 1 for RGB images

    Returns
    -------
    Image with rescaled intensity for imshow
    """
    image = zero_one_scale(image.clip(*range))
    return image


def check_circle_overlap(y1, x1, y2, x2, squared_radius, overlap_factor):
    """
    Checks if circles overlap, given coordinates and radius.
    Coordinates are expected in the same matrix notation as given by skimage.
    Function optimized for single-radius.

    Parameters
    ----------
    y1, x1:
        Center and radius of circle 1
    y2, x2:
        Center and radius of circle 2
    radius:
        Radius of circles
    overlap_factor:
        Kind of arbitrary number between 1 and 100 to decide degree of overlap. Around 20% works fine.

    Returns
    -------
    True and distance if circles overlap
    False and None, if circles don't overlap
    """

    distSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    radSumSq = (4 * squared_radius) / overlap_factor

    if distSq < radSumSq:
        return True, distSq
    else:
        return False, None


def light_blend(image1, image2, cmap1, cmap2):
    """
    Lighten blend of two images

    Parameters
    ----------
    image1:
        Image1 as single-channel
    image2:
        Image2 as single-channel
    cmap1:
        Colormap for image1
    cmap2:
        Colormap for image2

    Returns
    -------
    Blended image that can be displayed using plt.imshow()
    """
    a = cmap1(image1)
    b = cmap2(image2)

    screen = 1 - (1 - a) * (1 - b)
    return screen


def add_blend(image1, image2, blend_degree):
    """
    Additive blending of two images

    Parameters
    ----------
    image1:
        Image1 as RGBA
    image2:
        Image2 as RGBA
    blend_degree:
        Degree of blending, between 0 and 1

    Returns
    -------
    Blended image that can be displayed using plt.imshow()
    """
    return np.sqrt((1 - blend_degree) * image1 ** 2 + blend_degree * image2 ** 2)


def overlay_blend(image1, image2):
    """
    Overlay light_blend of two images

    Parameters
    ----------
    image1:
        Image1 as RGBA
    image2:
        Image2 as RGBA

    Returns
    -------
    Blended image that can be displayed using plt.imshow()
    """
    blend = np.choose(image1 > 0.5, [2 * image1 * image2, 1 - 2 * (1 - image1) * (1 - image2)])  # If False  # If True

    return blend


def soft_blend(image1, image2):
    """
    Soft light light_blend of two images

    Parameters
    ----------
    image1:
        Image1 as RGBA
    image2:
        Image2 as RGBA

    Returns
    -------
    Blended image that can be displayed using plt.imshow()
    """
    return (1 - 2 * image2) * image1 ** 2 + 2 * image1 * image2


def find_spots(image, value, method= "laplacian_of_gaussian") -> np.ndarray:
    """

    Parameters
    ----------
    image:
        input image
    value:
        value obtained from the user interface to be plugged in as either e.g. threshold value or number of peaks,
        depending on the specific method used
    method:
        method to detect particles

    Returns
    -------
    A list of (y, x) (<-- not a typo) tuples for the coordinates of each spot
    """
    if method == "peak_local_max":
        value = int(value)
        spots_found = skimage.feature.peak_local_max(
            image, min_distance=3, exclude_border=True, num_peaks=value
        )

    elif method == "laplacian_of_gaussian":
        spots_found = skimage.feature.blob_log(
            image, min_sigma=1.5, max_sigma=3, num_sigma=10, overlap=0.5, threshold=value
        )

    else:
        raise ValueError("Invalid method. Must be either 'peak_local_max' or 'laplacian_of_gaussian'.")

    if spots_found.shape[1] == 3:
        spots_found = spots_found[:, range(2)].tolist()
    else:
        spots_found = spots_found.tolist()

    return spots_found


def colocalize_rois(c1, c2, color1, color2, tolerance=1, math_radius=gvars.roi_math_radius) -> pd.DataFrame:
    """
    Checks if circles overlap, given coordinates and radius.
    Coordinates are expected in the same matrix notation as given by skimage
    Function optimized for single-radius

    Parameters
    ----------
    c1, c2:
        ndarray of circle centers as (y, x) returned by skimage
    math_radius:
        Single radius for all circles, used for calculation (not drawing)
    tolerance:
        Stringency of overlap.
    color1, color2:
        Names of the channels being colocalized (important for downstream colocalization of *all* channels)

    Returns
    -------
    Dataframe with coordinates of unique overlaps
    """

    all_combinations = np.array(list((itertools.product(c1, c2))))
    grid = np.hstack((all_combinations[:, 0], all_combinations[:, 1]))

    squared_radius = math_radius ** 2

    y1 = grid[:, 0]
    x1 = grid[:, 1]

    y2 = grid[:, 2]
    x2 = grid[:, 3]

    # Check if circles overlap
    distSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    radSumSq = (4 * squared_radius) / tolerance
    overlap = distSq < radSumSq

    # Keep only entries where they overlap
    grid = grid[overlap == True]
    distSq = distSq[overlap == True]

    grid = np.hstack((grid, distSq.reshape(-1, 1)))

    df = pd.DataFrame(
        grid,
        columns=("y_{}".format(color1), "x_{}".format(color1), "y_{}".format(color2), "x_{}".format(color2), "dist"),
    )

    df = (
        df.sort_values("dist")
        .drop_duplicates(("y_{}".format(color1), "x_{}".format(color1)))
        .drop_duplicates(("y_{}".format(color2), "x_{}".format(color2)))
        .reset_index(drop=True)
        .drop("dist", axis=1)
    )

    return df


def colocalize_triple(cpair1, cpair2, common_pair="green") -> pd.DataFrame:
    """
    Coocalizes

    Parameters
    ----------
    cpair1, cpair2:
        Pandas dataframes of colocalized channel pairs (e.g. blue/green, green/red)

    Returns
    -------
    Coordinates of spots that are colocalized in all 3 channels
    """

    if any(type(o) != pd.DataFrame for o in (cpair1, cpair2)):
        raise ValueError("Both inputs must be of type DataFrame")

    df = pd.merge(cpair1, cpair2, on=["y_{}".format(common_pair), "x_{}".format(common_pair)]).reset_index(drop=True)

    return df


def coloc_fraction(green, red, bluegreen, bluered, greenred, all):
    """
    Calculates the colocalization fraction of blue (external label) to FRET (green/red)-labelled proteins.
    This is calculated as fraction of blue that colocalizes with either green or red out of the total number of green/red

    Example
    -------
    labelled is 5, and total visible is 9, so coloc % is 5/9 = 55%

    labelled: 4 redblue + 3 greenblue - 2 all coloc
    total visible: 8 red + 3 green - 2 green/red

    RGB = red, green, blue

    Given the following number of spots/colocalizations:
    (R_B) (RGB) (_GB) (__B) (__B)
    (R_B) (RGB) (__B) (__B) (__B)
    (R__) (R__) (R__) (R__)

      labelled               total visible
    ( R_B + _GB - RGB )  / ( R + G - RG_ ) = %
    (4 + 3 - 2)          / (8 + 3 - 2)     = 0.55
    """
    return (bluered + bluegreen - all) / (red + green - greenred)


def circle_mask(inner_area, outer_area, gap_space, yx, indices):
    """
    Calculates a circular pixel mask for extracting get_intensities
    
    Parameters
    ----------
    inner_area:
        Area of the inner ROI
    outer_area:
        Area of ROI + background + space
    gap_space:
        Area of a circle
    yx:
        Coordinates in the format (y, x) (i.e. matrix indices row/col)
    indices:
        Image indices (obtained from np.indices(img.shape))

    Returns
    -------
    Center and background ring ROIs
    """

    yy, xx = yx

    yi, xi = indices
    mask = (yy - yi) ** 2 + (xx - xi) ** 2

    center = mask <= inner_area ** 2
    gap = mask <= inner_area ** 2 + gap_space ** 2
    bg_filled = mask <= outer_area ** 2
    bg_ring = np.logical_xor(bg_filled, gap)  # subtract inner circle_overlap from outer

    return center, bg_ring


def tiff_stack_intensity(array, roi_mask, bg_mask, raw=True):
    """
    Extracts get_intensities from TIFF stack, given ROI and BG masks.
    Intensities are calculated as medians of all pixel values within the ROIs.

    Parameters
    ----------
    array:
        Single-channel currentMovie array
    roi_mask:
        Numpy mask for center
    bg_mask:
        Numpy mask for background
    raw:
        Whether to return raw signal/background get_intensities. Otherwise will return signal-background and background as zeroes.

    Returns
    -------
    Center and background get_intensities
    """
    if len(array.shape) == 3:  # tiff-stack
        roi_pixel_sum_intensity = np.sum(array[:, roi_mask], axis=1)
        per_pixel_bg_intensity = np.median(array[:, bg_mask], axis=1)
        roi_n_pixels = len(roi_mask[roi_mask == True])

        bg_pixel_sum_intensity = per_pixel_bg_intensity * roi_n_pixels

    elif len(array.shape) == 2:  # single-frame tiff
        roi_pixel_sum_intensity = np.sum(array[roi_mask])
        per_pixel_bg_intensity = np.median(array[bg_mask])

        roi_n_pixels = len(roi_mask[roi_mask == True])
        bg_pixel_sum_intensity = per_pixel_bg_intensity * roi_n_pixels

    else:
        raise ValueError("Image format not recognized")

    if raw:
        return roi_pixel_sum_intensity, bg_pixel_sum_intensity
    else:
        signal = roi_pixel_sum_intensity - bg_pixel_sum_intensity

        background = np.zeros(signal.size)
        return signal, background
