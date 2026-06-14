"""
Fast image dehazing using guided joint bilateral filter — algorithm module.

Based on Xiao & Gan (Vis Comput, 2012). The method uses **two bilateral filters**
with different roles:

    1. Standard bilateral filter (``bilat_filter``)
       Input: dark-channel map W.
       Output: reference image R.
       Purpose: smooth texture in W while keeping depth edges, so R encodes
       where scene depth changes abruptly (object boundaries).

    2. Guided joint bilateral filter (``bilat_filter_joint``)
       Input: coarse atmospheric veil V, guided by reference R.
       Output: refined veil V_R.
       Purpose: smooth V, remove redundant texture, and restore depth jumps at
       edges using R. V_R is the transmission-related haze map used for recovery.

Full pipeline:
    1. W = min(B, G, R) — dark-channel proxy of the hazy image
    2. V = median-based coarse atmospheric veil (Tarel et al., Eq. 2)
    3. R = bilat_filter(W) — edge-preserving reference (paper Eq. 6)
    4. V_R = bilat_filter_joint(V, R) — refined veil (paper Eq. 7)
    5. A = estimate_atmospheric_light(I, W) — global airlight
    6. t = 1 - w * V_R / A — transmission map
    7. J = (I - A) / max(t, t0) + A — recovered scene radiance
    8. Optional LAB adaptive histogram equalization on L
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure


def mouse_callback(event, x, y, flags, param):
    """Close OpenCV windows on right-click."""
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyAllWindows()


def disp_img(img, title="img"):
    """Display an image in a separate OpenCV window (intermediate maps)."""
    display = img.copy()
    if display.dtype != np.uint8:
        peak = display.max()
        if peak > 0:
            display = (display / peak * 255).astype(np.uint8)
        else:
            display = display.astype(np.uint8)

    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    cv2.imshow(title, display)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(title, mouse_callback)


def _to_uint8_gray(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def adaptive_median_window(height, width):
    """
    Median-filter window size for the coarse veil V.

    Scales with image resolution so small images use smaller windows and
    large images avoid over-blurring depth edges.
    """
    size = int(np.floor(2 * max(height, width) / 50)) + 1
    return size if size % 2 == 1 else size + 1


def estimate_atmospheric_light(image_bgr, dark_channel):
    """
    Estimate global atmospheric light A (Sec. 3.2, He et al. style).

    Pick the brightest 0.2% dark-channel pixels, then take the brightest
    RGB intensity among those locations in the original hazy image.
    """
    num_pixels = max(int(np.floor(0.002 * dark_channel.size)), 1)
    flat_dark = dark_channel.ravel()
    top_idx = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    pixels = image_bgr.reshape(-1, 3).astype(np.float64)[top_idx]
    return float(np.max(pixels))


def _median_filter_symmetric(image, kernel_size):
    """
    Median filter with symmetric padding.

    Used in step 2 to build the coarse atmospheric veil V from W (Tarel et al.).
    """
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode="symmetric")
    filtered = ndimage.median_filter(padded, size=kernel_size, mode="nearest")
    return filtered[pad:-pad, pad:-pad]


def bilat_filter(input_image, radius, sigma_s, sigma_r, progress_callback=None):
    """
    Standard bilateral filter — first of the two bilateral stages (paper Eq. 3/6).

    Applied to the dark-channel map W to produce the reference image R.

    Purpose
    -------
    Edge-preserving smoothing of W: removes fine texture (brick patterns,
    foliage detail) but keeps large depth discontinuities. R is not used
    directly for dehazing; it guides the second filter so V_R respects
    object boundaries and avoids white fog halos around objects.

    Parameters
    ----------
    input_image : 2D array
        Typically W (min RGB channel).
    radius : int
        Spatial window half-size in pixels (GUI "Kernel").
    sigma_s : float
        Spatial Gaussian width; default 0.03 * min(height, width).
    sigma_r : float
        Range Gaussian width on intensity differences (GUI "Sigma_r").
    """
    rows, cols = input_image.shape
    output_image = np.zeros_like(input_image, dtype=np.float64)

    y_coords, x_coords = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )
    spatial_weights = np.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * sigma_s ** 2))

    for i in range(rows):
        row_min = max(i - radius, 0)
        row_max = min(i + radius, rows - 1)
        for j in range(cols):
            col_min = max(j - radius, 0)
            col_max = min(j + radius, cols - 1)

            local_input = input_image[row_min:row_max + 1, col_min:col_max + 1]
            range_weights = np.exp(
                -(local_input - input_image[i, j]) ** 2 / (2 * sigma_r ** 2)
            )
            sw = spatial_weights[
                (row_min - i + radius):(row_max - i + radius + 1),
                (col_min - j + radius):(col_max - j + radius + 1),
            ]
            weights = sw * range_weights
            output_image[i, j] = np.sum(weights * local_input) / np.sum(weights)

        if progress_callback is not None:
            progress_callback((i + 1) / rows)

    return output_image


def bilat_filter_joint(
    input_image, guidance_image, radius, sigma_s, sigma_r, sigma_t,
    progress_callback=None,
):
    """
    Guided joint bilateral filter — second bilateral stage (paper Eq. 7).

    Filters the coarse atmospheric veil V while using reference R as guidance,
    producing the refined veil V_R used in the transmission map.

    Purpose
    -------
    Median filtering (step 2) yields a smooth but edge-blurred V. This filter:
      - keeps V smooth in regions of similar depth (spatial term f),
      - follows depth edges from R via g(R(x) - R(y)),
      - suppresses texture inconsistent with R via h(V(y) - R(y)).

    That combination removes leftover haze texture and restores sharp depth
    jumps, which is critical for correct transmission near object outlines.

    Parameters
    ----------
    input_image : 2D array
        Coarse atmospheric veil V.
    guidance_image : 2D array
        Reference R from ``bilat_filter(W, ...)``.
    radius, sigma_s : same as ``bilat_filter``.
    sigma_r : float
        Range width for guidance term g on R (GUI "Sigma_r").
    sigma_t : float
        Range width for correction term h on V(y) - R(y) (GUI "Sigma_t").
    """
    rows, cols = input_image.shape
    output_image = np.zeros_like(input_image, dtype=np.float64)

    y_coords, x_coords = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )
    spatial_weights = np.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * sigma_s ** 2))

    for i in range(rows):
        row_min = max(i - radius, 0)
        row_max = min(i + radius, rows - 1)
        for j in range(cols):
            col_min = max(j - radius, 0)
            col_max = min(j + radius, cols - 1)

            local_input = input_image[row_min:row_max + 1, col_min:col_max + 1]
            local_guidance = guidance_image[row_min:row_max + 1, col_min:col_max + 1]

            guidance_weights = np.exp(
                -(local_guidance - guidance_image[i, j]) ** 2 / (2 * sigma_r ** 2)
            )
            correction_weights = np.exp(
                -(local_input - local_guidance) ** 2 / (2 * sigma_t ** 2)
            )
            sw = spatial_weights[
                (row_min - i + radius):(row_max - i + radius + 1),
                (col_min - j + radius):(col_max - j + radius + 1),
            ]
            weights = sw * guidance_weights * correction_weights
            output_image[i, j] = np.sum(weights * local_input) / np.sum(weights)

        if progress_callback is not None:
            progress_callback((i + 1) / rows)

    return output_image


def lab_adapthisteq(image_bgr):
    """
    Optional post-processing: CLAHE on the L channel in LAB space.

    Brightens local contrast after dehazing; not part of the core paper method.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64) / 255.0
    l_eq = exposure.equalize_adapthist(
        l_channel, kernel_size=(8, 8), clip_limit=0.005,
    )
    lab[:, :, 0] = np.clip(l_eq * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def dehaze(
    image,
    show_w=False,
    show_v=False,
    show_r=False,
    show_v_r=False,
    show_t=False,
    show_j=False,
    adapt_eq=False,
    progress_callback=None,
    radius=15,
    omega=None,
    sigma_s=None,
    sigma_r=20.0,
    sigma_t=20.0,
    p=0.95,
    w=0.95,
    t0=0.3,
):
    """
    Run the full dehazing pipeline (two bilateral filters + atmospheric model).

    Intermediate maps (GUI preview switches)
    ----------------------------------------
    W   : min(R,G,B) — dark-channel input to the pipeline.
    V   : coarse atmospheric veil after median filtering.
    R   : reference image after the **first** bilateral filter on W.
    V_R : refined veil after the **second** (guided joint) bilateral filter.
    t   : transmission map derived from V_R and airlight A.
    J   : recovered haze-free image.

    Key parameters
    --------------
    radius / omega : bilateral kernel size and median window (odd integers).
    sigma_r, sigma_t : range widths for the two bilateral stages (floats).
    w : haze retention in transmission (paper omega, default 0.95).
    p : coarse-veil strength from median step (default 0.95).
    t0 : minimum transmission to limit noise amplification (default 0.3).

    Returns
    -------
    dict with keys J, J_adapt, W, V, R, V_R, t, A.
    """
    height, width = image.shape[:2]
    if sigma_s is None:
        sigma_s = 0.03 * min(height, width)
    if omega is None:
        omega = adaptive_median_window(height, width)

    image_double = image.astype(np.float64)
    W = np.min(image_double, axis=2)

    B = _median_filter_symmetric(W, omega)
    C = B - _median_filter_symmetric(np.abs(W - B), omega)
    V = np.maximum(np.minimum(p * C, W), 0)

    def _progress(stage_frac, row_frac):
        if progress_callback is not None:
            progress_callback(0.5 * stage_frac + 0.5 * row_frac)

    r_progress = (lambda f: _progress(0, f)) if progress_callback else None
    vr_progress = (lambda f: _progress(1, f)) if progress_callback else None

    R = bilat_filter(W, radius, sigma_s, sigma_r, progress_callback=r_progress) # same as cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType=cv2.BORDER_DEFAULT)
    V_R = bilat_filter_joint(
        V, R, radius, sigma_s, sigma_r, sigma_t, progress_callback=vr_progress,
    )

    A = max(estimate_atmospheric_light(image, W), 1.0)
    t = np.clip(1.0 - w * V_R / A, t0, 1.0)

    J = np.zeros_like(image_double)
    t_safe = t
    for c in range(3):
        J[:, :, c] = (image_double[:, :, c] - A) / t_safe + A
    J_uint8 = np.clip(J, 0, 255).astype(np.uint8)

    J_adapt = None
    if adapt_eq:
        J_adapt = lab_adapthisteq(J_uint8)

    if show_w:
        disp_img(_to_uint8_gray(W), title="W")
    if show_v:
        disp_img(_to_uint8_gray(V), title="V")
    if show_r:
        disp_img(_to_uint8_gray(R), title="R")
    if show_v_r:
        disp_img(_to_uint8_gray(V_R), title="V_R")
    if show_t:
        disp_img(t, title="depth image t")
    if show_j:
        disp_img(J_uint8, title="J")
        disp_img(image, title="Original")

    if adapt_eq and J_adapt is not None:
        disp_img(J_adapt, title="J_adaptEQ")

    return {
        "J": J_uint8,
        "J_adapt": J_adapt,
        "W": W,
        "V": V,
        "R": R,
        "V_R": V_R,
        "t": t,
        "A": A,
    }
