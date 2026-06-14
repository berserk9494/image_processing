"""
Homomorphic image dehazing — algorithm module.

Pipeline:
    1. Grayscale conversion
    2. Build frequency-domain high-pass filter H (GausHP / ButterHP / IdealHP)
    3. log(1 + I) → FFT → multiply by H → IFFT → exp() - 1
    4. Scale each RGB channel by the dehazed gray result
    5. Optional LAB adaptive histogram equalization on L
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from skimage import exposure


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyAllWindows()


def disp_img(img, title="img"):
    """Display a grayscale or BGR image in an OpenCV window."""
    display = img.copy()
    if display.dtype != np.uint8:
        if display.ndim == 2:
            peak = np.max(np.abs(display))
            if peak > 0:
                display = (display / peak * 255).astype(np.uint8)
            else:
                display = np.clip(display * 255, 0, 255).astype(np.uint8)
        else:
            display = np.clip(display * 255 if display.max() <= 1.0 else display, 0, 255).astype(np.uint8)

    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    cv2.imshow(title, display)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(title, mouse_callback)


def spectrum_to_uint8(spectrum):
    """
    Convert a complex FFT array to a viewable grayscale image.

    Linear scaling by max(|spectrum|) hides almost all energy because the DC
    bin is orders of magnitude brighter than the rest. Log compression matches
    how FFT spectra are usually displayed (and what MATLAB imshow effectively
    reveals for frequency data).
    """
    magnitude = np.log1p(np.abs(spectrum))
    lo, hi = magnitude.min(), magnitude.max()
    if hi > lo:
        magnitude = (magnitude - lo) / (hi - lo) * 255.0
    return magnitude.astype(np.uint8)


def disp_spectrum(spectrum, title="spectrum"):
    """Display log-compressed magnitude of a complex FFT array."""
    disp_img(spectrum_to_uint8(spectrum), title=title)


def _freq_mesh(shape):
    """MATLAB meshgrid(1:N, 1:M) frequency coordinates."""
    m, n = shape
    x, y = np.meshgrid(np.arange(1, n + 1), np.arange(1, m + 1))
    cx, cy = np.floor(n / 2), np.floor(m / 2)
    return x, y, cx, cy


def gaushp(shape, g_l, g_h, d0, c):
    """Gaussian homomorphic high-pass filter."""
    x, y, cx, cy = _freq_mesh(shape)
    gaussian_numerator = (x - cx) ** 2 + (y - cy) ** 2
    return (g_h - g_l) * (1 - np.exp(-c * gaussian_numerator / (d0 ** 2))) + g_l


def butterhp(shape, d0, n):
    """Butterworth high-pass filter."""
    x, y, cx, cy = _freq_mesh(shape)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = 1.0 / (1.0 + (d0 / d) ** (2 * n))
    h[d == 0] = 0.0
    return h


def idealhp(shape, cutoff):
    """Ideal high-pass filter (squared-distance cutoff)."""
    x, y, cx, cy = _freq_mesh(shape)
    d_sq = (x - cx) ** 2 + (y - cy) ** 2
    return (d_sq >= cutoff).astype(np.float64)


def homomorphic_filter(gray, h):
    """
    Apply homomorphic filtering.

    Returns dehazed gray, shifted FFT of log image, and filtered spectrum.
    """
    i_log = np.log1p(gray)
    i_fft = np.fft.fftshift(np.fft.fft2(i_log))
    g = h * i_fft
    i_out = np.real(np.fft.ifft2(np.fft.ifftshift(g)))
    i_out = np.expm1(i_out)
    return i_out, i_fft, g


def lab_adapthisteq(image_bgr_uint8):
    """LAB adaptive histogram equalization on the L channel."""
    lab = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64) / 255.0
    l_eq = exposure.equalize_adapthist(
        l_channel, kernel_size=(8, 8), clip_limit=0.005,
    )
    lab[:, :, 0] = np.clip(l_eq * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def build_filter(filter_type, shape, **params):
    """Build H for the selected filter type."""
    if filter_type == "gaushp":
        return gaushp(shape, params["g_l"], params["g_h"], params["d0"], params["c"])
    if filter_type == "butterhp":
        return butterhp(shape, params["d0"], int(params["n"]))
    if filter_type == "idealhp":
        return idealhp(shape, int(params["d0"]))
    raise ValueError(f"Unknown filter type: {filter_type}")


def homomorphic_dehaze(
    image_bgr,
    h,
    show_i_fft=False,
    show_g_fft=False,
    show_dehazed_gray=False,
    show_dehazed_rgb=False,
    adapt_eq=False,
):
    """
    Run the full homomorphic dehazing pipeline.

    Parameters
    ----------
    image_bgr : ndarray
        BGR uint8 image.
    h : ndarray
        Frequency-domain filter (same size as grayscale image).

    Returns
    -------
    dict with I_defog (uint8 BGR), I_gray_defog, If, G, J_adapt.
    """
    image_float = image_bgr.astype(np.float64) / 255.0
    i_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    i_gray_defog, i_fft, g = homomorphic_filter(i_gray, h)

    i_defog = np.zeros_like(image_float)
    for c in range(3):
        i_defog[:, :, c] = image_float[:, :, c] * i_gray_defog

    i_defog_uint8 = np.clip(i_defog * 255, 0, 255).astype(np.uint8)

    j_adapt = None
    if adapt_eq:
        j_adapt = lab_adapthisteq(i_defog_uint8)

    if show_i_fft:
        disp_spectrum(i_fft, title="I_FFT")
    if show_g_fft:
        disp_spectrum(g, title="G*I_FFT")
    if show_dehazed_gray:
        disp_img(i_gray_defog, title="Dehazed Gray")
    if show_dehazed_rgb:
        disp_img(i_defog_uint8, title="Dehazed")
    if adapt_eq and j_adapt is not None:
        disp_img(j_adapt, title="J_adaptEQ")

    return {
        "I_defog": i_defog_uint8,
        "I_gray_defog": i_gray_defog,
        "If": i_fft,
        "G": g,
        "J_adapt": j_adapt,
    }


def _filter_center(shape):
    m, n = shape
    return int(np.floor(m / 2)), int(np.floor(n / 2))


def _adaptive_crop_radius(shape, filter_type, params):
    """Pick a zoom radius where the filter transition is visible."""
    m, n = shape
    max_r = min(m, n) // 2 - 1
    max_r = max(max_r, 1)
    params = params or {}

    if filter_type == "gaushp":
        d0 = max(float(params.get("d0", 1)), 1.0)
        c = max(float(params.get("c", 2)), 0.1)
        # Transition scale in pixel-index units
        radius = int(max(d0 * np.sqrt(max(c, 1)) * 4, 25))
    elif filter_type == "butterhp":
        d0 = max(float(params.get("d0", 1)), 1.0)
        radius = int(max(d0 * 4, 25))
    else:
        cutoff = max(float(params.get("d0", 1)), 1.0)
        radius = int(max(np.sqrt(cutoff) * 3, 25))

    return min(radius, max_r, 150)


def _radial_profile(h):
    """Average H over concentric rings (matches MATLAB frequency grid)."""
    m, n = h.shape
    x, y = np.meshgrid(np.arange(1, n + 1), np.arange(1, m + 1))
    cx, cy = np.floor(n / 2), np.floor(m / 2)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = int(min(float(r.max()), min(m, n) // 2))
    radii = np.arange(0, max_r + 1)
    profile = np.empty_like(radii, dtype=np.float64)
    for i, ri in enumerate(radii):
        if ri == 0:
            mask = r < 0.5
        else:
            mask = (r >= ri - 0.5) & (r < ri + 0.5)
        profile[i] = h[mask].mean() if mask.any() else np.nan
    return radii, profile


def plot_transfer_function(h, filter_type="", params=None):
    """
    Show the frequency-domain filter clearly:
      - 2D heatmap (adaptive center zoom)
      - radial profile H vs distance from DC
      - 3D surface of the zoomed region
    """
    params = params or {}
    m, n = h.shape
    cy, cx = _filter_center(h.shape)
    crop_r = _adaptive_crop_radius(h.shape, filter_type, params)

    y0, y1 = cy - crop_r, cy + crop_r + 1
    x0, x1 = cx - crop_r, cx + crop_r + 1
    crop = h[y0:y1, x0:x1]

    h_min, h_max = float(h.min()), float(h.max())
    if h_max <= h_min:
        h_max = h_min + 1.0

    fig = plt.figure(figsize=(13, 4.5))

    ax1 = fig.add_subplot(131)
    im = ax1.imshow(
        crop, cmap="gray", vmin=h_min, vmax=h_max,
        interpolation="bilinear", origin="upper",
    )
    ax1.set_title(f"H (u,v) — zoom ±{crop_r} px")
    ax1.set_xlabel("u")
    ax1.set_ylabel("v")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label("Gain")

    ax2 = fig.add_subplot(132)
    radii, profile = _radial_profile(h)
    ax2.plot(radii, profile, color="#0078d4", linewidth=1.8)
    ax2.axhline(h_min, color="#888888", linestyle="--", linewidth=0.8, label=f"min={h_min:.3g}")
    ax2.axhline(h_max, color="#444444", linestyle="--", linewidth=0.8, label=f"max={h_max:.3g}")
    ax2.set_xlim(0, min(len(radii) - 1, crop_r * 3))
    ax2.set_ylim(h_min - 0.05 * (h_max - h_min), h_max + 0.05 * (h_max - h_min))
    ax2.set_xlabel("Distance from DC")
    ax2.set_ylabel("H")
    ax2.set_title("Radial profile")
    ax2.grid(True, alpha=0.35)
    ax2.legend(fontsize=8, loc="best")

    ax3 = fig.add_subplot(133, projection="3d")
    step = max(1, crop.shape[0] // 80, crop.shape[1] // 80)
    surf = crop[::step, ::step]
    yy, xx = np.meshgrid(np.arange(surf.shape[1]), np.arange(surf.shape[0]))
    ax3.plot_surface(
        xx, yy, surf, cmap="gray", alpha=0.95,
        linewidth=0, antialiased=True, vmin=h_min, vmax=h_max,
    )
    ax3.set_xlabel("u")
    ax3.set_ylabel("v")
    ax3.set_zlabel("H")
    ax3.set_title("3D surface")
    ax3.view_init(elev=32, azim=-58)

    label = {"gaushp": "GausHP", "butterhp": "ButterHP", "idealhp": "IdealHP"}.get(
        filter_type, filter_type or "Filter",
    )
    if params:
        param_str = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        )
        fig.suptitle(f"{label} transfer function  ({param_str})", fontsize=12, y=1.0)
    else:
        fig.suptitle(f"{label} transfer function", fontsize=12, y=1.0)

    fig.tight_layout()


def show_histogram(gray_float):
    """Show grayscale histogram in a matplotlib figure."""
    plt.figure("Histogram")
    plt.hist(gray_float.ravel(), bins=256, range=(0, 1), color="steelblue")
    plt.title("Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
