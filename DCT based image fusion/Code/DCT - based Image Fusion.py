"""
Discrete Cosine Transform-based Multi-Focus Image Fusion.

Merges two partially-focused grayscale images into a single all-in-focus result
using multi-resolution DCT (MDCT) decomposition. A DWT baseline is included for
comparison. See Code/Discrete_Cosine_Transform-based_Image_Fusion.pdf for the
full method description.

Pipeline:
    1. MDCT/MDWT decompose each source into LL (approximation) and LH/HL/HH (details).
    2. Fuse subbands with max, min, or mean absolute-magnitude rules.
    3. IMDCT/IMDWT reconstruct the fused image.

Example test images: Images/img1.png, Images/img2.png, Images/Reference.png

Run:
    python "DCT - based Image Fusion.py"
"""

import tkinter as tk
import tkinter.ttk
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.fft import dct, idct
from tabulate import tabulate

# Loaded images for the GUI: [source_1, source_2] or [source_1, source_2, reference].
images = []

# Standard DCT/IDCT settings used throughout MDCT and IMDCT.
_DCT_TYPE = 2
_DCT_NORM = "ortho"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def mouse_callback(event, x, y, flags, param):
    """Close OpenCV windows on right-click."""
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyAllWindows()


def create_mosaic(subbands):
    """
    Build a 2x2 mosaic of subbands for visualization.

    Layout (wavelet-style):
        [LL | HL]
        [LH | HH]

    For multiple decomposition levels, higher-level LL mosaics are placed in the
    top-left quadrant recursively.
    """
    def img_norm(img):
        peak = img.max()
        return img / peak if peak != 0 else img

    if len(subbands) == 1:
        return subbands[0]

    if len(subbands) <= 4:
        return np.block([
            [img_norm(subbands[0]), img_norm(subbands[2])],
            [img_norm(subbands[1]), img_norm(subbands[3])],
        ])

    return np.block([
        [img_norm(create_mosaic(subbands[4:])), img_norm(subbands[2])],
        [img_norm(subbands[1]), img_norm(subbands[3])],
    ])


# ---------------------------------------------------------------------------
# Fusion coefficient selection (shared by DCT and DWT paths)
# ---------------------------------------------------------------------------

def get_max_magnitude_pixel(A, B):
    """
    Per-pixel selection with the larger absolute value.

    Ties (equal magnitude) keep the value from image B.
    """
    assert A.shape == B.shape
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    higher_mag_mask = np.abs(A) > np.abs(B)
    result = np.zeros_like(A)
    result[higher_mag_mask] = A[higher_mag_mask]
    result[~higher_mag_mask] = B[~higher_mag_mask]
    return result


def get_min_magnitude_pixel(A, B):
    """
    Per-pixel selection with the smaller absolute value.

    Ties (equal magnitude) keep the value from image B.
    """
    assert A.shape == B.shape
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    lower_mag_mask = np.abs(A) < np.abs(B)
    result = np.zeros_like(A)
    result[lower_mag_mask] = A[lower_mag_mask]
    result[~lower_mag_mask] = B[~lower_mag_mask]
    return result


def _fuse_subbands(coeffs_1, coeffs_2, details, approx):
    """
    Apply fusion rules to aligned subband lists.

    LH/HL/HH are fused at every level. LL is fused only at the deepest level
    (index len-4); shallow LL bands are placeholders because IMDCT/IMDWT replace
    them with the coarser reconstructed image during assembly.
    """
    deepest_ll_index = len(coeffs_1) - 4
    fused = []
    for i in range(len(coeffs_1)):
        c1, c2 = coeffs_1[i], coeffs_2[i]
        if i % 4 == 0:
            if i == deepest_ll_index:
                if approx == "min":
                    fused.append(np.where(np.abs(c1) < np.abs(c2), c1, c2)) # get_min_magnitude_pixel(c1, c2)
                elif approx == "max":
                    fused.append(np.where(np.abs(c1) > np.abs(c2), c1, c2)) # get_max_magnitude_pixel(c1, c2)
                else:
                    fused.append((c1 + c2) / 2) # np.mean([c1, c2], axis=0)
            else:
                fused.append(c1)
        else:
            if details == "min":
                fused.append(np.where(np.abs(c1) < np.abs(c2), c1, c2)) # get_min_magnitude_pixel(c1, c2)
            elif details == "max":
                fused.append(np.where(np.abs(c1) > np.abs(c2), c1, c2)) # get_max_magnitude_pixel(c1, c2)
            elif details == "mean":
                fused.append((c1 + c2) / 2) # np.mean([c1, c2], axis=0)
            else:
                raise ValueError(f"Invalid details rule: {details}")
    return fused


# ---------------------------------------------------------------------------
# Multi-resolution DCT (MDCT) — primary method from the article
# ---------------------------------------------------------------------------

def MDCT(image, level=1):
    """
    Multi-resolution DCT decomposition (MDCT).

    Each level splits the current approximation band into four subbands:
    LL (low-low), LH (low-high), HL (high-low), HH (high-high), using
    separable 2-D DCT followed by quadrant splitting in the transform domain.

    Parameters
    ----------
    image : ndarray
        2-D grayscale image (float or integer).
    level : int
        Number of decomposition levels.

    Returns
    -------
    list of ndarray
        Flat list [LL_1, LH_1, HL_1, HH_1, LL_2, ...] per level.
    """
    coeffs = []

    for _ in range(level):
        # Step 1: column-wise DCT. (colapse rows into columns)
        dct_cols = dct(image, axis=0, type=_DCT_TYPE, norm=_DCT_NORM)

        # Step 2: split rows into low and high frequency halves.
        split_row = dct_cols.shape[0] // 2
        upper_half = dct_cols[:split_row, :] # low frequency half
        lower_half = dct_cols[split_row:, :] # high frequency half

        # Step 3: partial inverse along columns to obtain separable components.
        idct_upper = idct(upper_half, axis=0, type=_DCT_TYPE, norm=_DCT_NORM)
        idct_lower = idct(lower_half, axis=0, type=_DCT_TYPE, norm=_DCT_NORM)

        # Step 4: row-wise DCT on each half.
        dct_rows_upper = dct(idct_upper, axis=1, type=_DCT_TYPE, norm=_DCT_NORM)
        dct_rows_lower = dct(idct_lower, axis=1, type=_DCT_TYPE, norm=_DCT_NORM)

        # Step 5: split columns into LL/LH and HL/HH, then inverse along rows.
        split_col_u = dct_rows_upper.shape[1] // 2
        split_col_l = dct_rows_lower.shape[1] // 2
        LL = idct(dct_rows_upper[:, :split_col_u], axis=1, type=_DCT_TYPE, norm=_DCT_NORM)
        LH = idct(dct_rows_upper[:, split_col_u:], axis=1, type=_DCT_TYPE, norm=_DCT_NORM)
        HL = idct(dct_rows_lower[:, :split_col_l], axis=1, type=_DCT_TYPE, norm=_DCT_NORM)
        HH = idct(dct_rows_lower[:, split_col_l:], axis=1, type=_DCT_TYPE, norm=_DCT_NORM)

        coeffs.extend([LL, LH, HL, HH])
        image = LL.copy() # used for the next level decomposition

    return coeffs


def IMDCT(coeff):
    """
    Inverse multi-resolution DCT reconstruction (IMDCT).

    Reverses MDCT level by level, starting from the deepest decomposition.
    """
    batches = [coeff[i:i + 4] for i in range(0, len(coeff), 4)]
    batches = batches[::-1]

    def assemble_level(LL_I, LH_I, HL_I, HH_I):
        LL = dct(LL_I, axis=1, type=_DCT_TYPE, norm=_DCT_NORM) # row-wise DCT
        LH = dct(LH_I, axis=1, type=_DCT_TYPE, norm=_DCT_NORM) # row-wise DCT
        HL = dct(HL_I, axis=1, type=_DCT_TYPE, norm=_DCT_NORM) # row-wise DCT
        HH = dct(HH_I, axis=1, type=_DCT_TYPE, norm=_DCT_NORM) # row-wise DCT
        block_up_col = idct(np.block([LL, LH]), axis=1, type=_DCT_TYPE, norm=_DCT_NORM) # row-wise IDCT
        block_low_col = idct(np.block([HL, HH]), axis=1, type=_DCT_TYPE, norm=_DCT_NORM) # row-wise IDCT
        block_up_row = dct(block_up_col, axis=0, type=_DCT_TYPE, norm=_DCT_NORM) # column-wise DCT
        block_low_row = dct(block_low_col, axis=0, type=_DCT_TYPE, norm=_DCT_NORM) # column-wise DCT
        return idct(np.block([[block_up_row], [block_low_row]]), axis=0, type=_DCT_TYPE, norm=_DCT_NORM) # column-wise IDCT

    reconstructed = []
    for i, batch in enumerate(batches):
        if i == 0:
            reconstructed.append(assemble_level(*batch))
        else:
            reconstructed.append(assemble_level(reconstructed[i - 1], batch[1], batch[2], batch[3]))

    return reconstructed[-1] if reconstructed else None


# ---------------------------------------------------------------------------
# Multi-resolution DWT — comparison baseline
# ---------------------------------------------------------------------------

def MDWT(image, level, wavelet):
    """Multi-resolution 2-D DWT; returns flat subband list like MDCT."""
    coeffs = []
    for _ in range(level):
        LL, (LH, HL, HH) = pywt.dwt2(image, wavelet)
        coeffs.extend([LL, LH, HL, HH])
        image = LL.copy()
    return coeffs


def IMDWT(coeff, wavelet):
    """Inverse multi-resolution DWT reconstruction."""
    batches = [coeff[i:i + 4] for i in range(0, len(coeff), 4)]
    batches = batches[::-1]

    reconstructed = []
    for i, batch in enumerate(batches):
        subbands = (batch[0] if i == 0 else reconstructed[i - 1], (batch[1], batch[2], batch[3]))
        reconstructed.append(pywt.idwt2(subbands, wavelet))

    return reconstructed[-1] if reconstructed else None


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _show_fusion_views(level, approx, details, source_mosaic_1, source_mosaic_2,
                       fusion_mosaic, fused_image, labels, err=False, wavelet=None):
    """Display optional OpenCV windows for decomposition and fusion results."""
    windows_shown = False

    if labels.get("decomposition"):
        combined = cv2.hconcat([source_mosaic_1, source_mosaic_2])
        cv2.putText(combined, labels["img1"], (220, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, labels["img2"], (source_mosaic_1.shape[1] + 220, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        title = f"Decomposition level: {level}"
        cv2.imshow(title, combined)
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(title, mouse_callback)
        windows_shown = True

    if labels.get("fusion_mosaic"):
        caption = f"Approx: {approx} Details: {details}"
        if wavelet:
            caption = f"Mother Wavelet: {wavelet} " + caption
        cv2.putText(fusion_mosaic, caption, (70, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        title = f"Decomposition of Fusion, level: {level}"
        cv2.imshow(title, fusion_mosaic)
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(title, mouse_callback)
        windows_shown = True

    if labels.get("fusion"):
        cv2.putText(fused_image, f"Approx: {approx} Details: {details}", (165, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("Fusion", fused_image.copy() / 255.0)
        cv2.setWindowProperty("Fusion", cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback("Fusion", mouse_callback)
        windows_shown = True

    if windows_shown and not err:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def fusion_dct(img_1, img_2, level, details, approx,
               display_decomposition_mosaic=False, display_fusion_mosaic=False,
               display_fusion=False, err=False):
    """Fuse two images using MDCT subband rules, then reconstruct with IMDCT."""
    coeffs_1 = MDCT(img_1, level)
    coeffs_2 = MDCT(img_2, level)
    coeff = _fuse_subbands(coeffs_1, coeffs_2, details, approx)

    mosaic_1 = create_mosaic(coeffs_1)
    mosaic_2 = create_mosaic(coeffs_2)
    fusion_mosaic = create_mosaic(coeff)
    fused_image = IMDCT(coeff)

    _show_fusion_views(
        level, approx, details, mosaic_1, mosaic_2, fusion_mosaic, fused_image,
        labels={
            "decomposition": display_decomposition_mosaic,
            "fusion_mosaic": display_fusion_mosaic,
            "fusion": display_fusion,
            "img1": "IDCT_IMG1",
            "img2": "IDCT_IMG2",
        },
        err=err,
    )

    return fused_image, coeff, mosaic_1, mosaic_2, fusion_mosaic


def fusion_dwt(img_1, img_2, level, wavelet, details, approx,
               display_decomposition_mosaic=False, display_fusion_mosaic=False,
               display_fusion=False, err=False):
    """Fuse two images using DWT subband rules (comparison baseline)."""
    coeffs_1 = MDWT(img_1, level, wavelet)
    coeffs_2 = MDWT(img_2, level, wavelet)
    coeff = _fuse_subbands(coeffs_1, coeffs_2, details, approx)

    mosaic_1 = create_mosaic(coeffs_1)
    mosaic_2 = create_mosaic(coeffs_2)
    fusion_mosaic = create_mosaic(coeff)
    fused_image = IMDWT(coeff, wavelet)

    _show_fusion_views(
        level, approx, details, mosaic_1, mosaic_2, fusion_mosaic, fused_image,
        labels={
            "decomposition": display_decomposition_mosaic,
            "fusion_mosaic": display_fusion_mosaic,
            "fusion": display_fusion,
            "img1": "IDWT_IMG1",
            "img2": "IDWT_IMG2",
        },
        err=err,
        wavelet=wavelet,
    )

    return fused_image, coeff, mosaic_1, mosaic_2, fusion_mosaic


# ---------------------------------------------------------------------------
# Quality metrics (article evaluation section)
# ---------------------------------------------------------------------------

def PFE(I_r, I_f):
    """Percent Fusion Error: 100 * ||I_r - I_f|| / ||I_r|| (lower is better)."""
    norm_I_r = np.linalg.norm(I_r)
    if norm_I_r == 0:
        return 0.0
    return round((np.linalg.norm(I_r - I_f) / norm_I_r) * 100, 4)


def PSNR(I_r, I_f):
    """Peak Signal-to-Noise Ratio in dB (higher is better)."""
    mse = np.mean((I_r - I_f) ** 2)
    if mse == 0:
        return float("inf")
    return round(10 * np.log10(255 ** 2 / mse), 4)


def SD(I_f):
    """Standard deviation of the fused image histogram (contrast measure)."""
    hist, bins = np.histogram(I_f.flatten(), bins=256)
    mean = np.sum(hist * bins[:-1]) / np.sum(hist)
    variance = np.sum((bins[:-1] - mean) ** 2 * hist) / np.sum(hist)
    return round(np.sqrt(variance), 4)


def MSSISM(I_r, I_f, L=255):
    """Mean Structural Similarity Index (SSIM); higher is better."""
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    mu1 = cv2.GaussianBlur(I_r, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I_f, (11, 11), 1.5)
    mu1_2, mu2_2, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I_r ** 2, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I_f ** 2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I_r * I_f, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2)
    return round(float(np.mean(numerator / denominator)), 4)


def CE(I1, I2, I_f):
    """Average cross-entropy between sources and fused image (lower is better)."""
    def calculate_pmf(image):
        hist, _ = np.histogram(image.flatten(), bins=256)
        return hist / np.sum(hist)

    def cross_entropy(Y, P):
        epsilon = 2 ** -32
        hist_y = np.clip(calculate_pmf(Y), epsilon, 1)
        hist_p = np.clip(calculate_pmf(P), epsilon, 1)
        return np.sum(hist_y * np.log(hist_y / hist_p))

    return round((cross_entropy(I1, I_f) + cross_entropy(I2, I_f)) * 0.5, 4)


def SF(I_f):
    """Spatial frequency — image sharpness measure (higher is better)."""
    def RF(X):
        diffs =np.diff(X, axis=1) ** 2 #[(X[i, j] - X[i, j - 1]) ** 2 for i in range(X.shape[0]) for j in range(1, X.shape[1])]
        return np.sum(diffs) / (X.shape[0] * X.shape[1])

    def CF(X):
        diffs = np.diff(X, axis=0) ** 2 #[(X[i, j] - X[i - 1, j]) ** 2 for i in range(1, X.shape[0]) for j in range(X.shape[1])]
        return np.sum(diffs) / (X.shape[0] * X.shape[1])

    return round(float(np.sqrt(RF(I_f) + CF(I_f))), 4)


def dict_to_table(data, fusion_method, level):
    """Print a formatted metric table to the console."""
    df = pd.DataFrame(data.items(), columns=[f"Fusion: {fusion_method} Level: {level}", "Metric Value"])
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


def fusion(image_list, level, fusion_method, wavelet, details, approx,
           display_decomposition_mosaic=False, display_fusion_mosaic=False,
           display_fusion=False, display_error=False, calculate_metrics=False):
    """
    Top-level fusion entry point used by the GUI and notebook.

    image_list: 2 images for fusion-only metrics, or 3 with a reference for PFE/PSNR/SSIM.
    """
    img_1, img_2 = image_list[0], image_list[1]
    reference_img = image_list[2] if len(image_list) > 2 else None

    if fusion_method == "DCT":
        out, _, _, _, _ = fusion_dct(
            img_1, img_2, level, details, approx,
            display_decomposition_mosaic, display_fusion_mosaic, display_fusion,
            err=display_error,
        )
    elif fusion_method == "DWT":
        out, _, _, _, _ = fusion_dwt(
            img_1, img_2, level, wavelet, details, approx,
            display_decomposition_mosaic, display_fusion_mosaic, display_fusion,
            err=display_error,
        )
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    if display_error and reference_img is not None:
        cv2.imshow("Error", (out - reference_img) / 255.0)
        cv2.setMouseCallback("Error", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if calculate_metrics:
        info_no_ref = {"CE": CE(img_1, img_2, out), "SD": SD(out), "SF": SF(out)}
        if reference_img is not None:
            info_ref = {
                "PFE": PFE(reference_img, out),
                "PSNR": PSNR(reference_img, out),
                "SSIM": MSSISM(reference_img, out, L=255),
            }
            info = {**info_ref, **info_no_ref}
        else:
            info = info_no_ref
        dict_to_table(info, fusion_method, level)
        return info

    return None


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def select_images_from_folder():
    """Open a file dialog and load up to 3 grayscale images resized to 512x512."""
    global images
    file_paths = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.tif;*.tiff")],
    )
    images = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: could not read image: {file_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512, 512))
        images.append(image.astype(np.float64))
    enable_btn()
    return images


def start_fusion(display_decomposition_mosaic=False, display_fusion_mosaic=False,
                 display_fusion=False, display_error=False, calculate_metrics=False):
    """Read GUI settings and run fusion with the selected display mode."""
    global images

    if not images or len(images) < 2:
        print("Select at least two images.")
        return

    info = fusion(
        images,
        int(decomposition_level_scale.get()),
        fusion_model_combo.get(),
        dwt_selected_combo.get(),
        details_combo.get(),
        approximation_combo.get(),
        display_decomposition_mosaic=display_decomposition_mosaic,
        display_fusion_mosaic=display_fusion_mosaic,
        display_fusion=display_fusion,
        display_error=display_error,
        calculate_metrics=calculate_metrics,
    )
    insert_data_to_metric_table(info)


def display_selected_images():
    """Show loaded source (and optional reference) images side by side."""
    global images
    if not images:
        print("No images selected.")
        return

    if len(images) > 2:
        preview = cv2.hconcat([images[0], images[1], images[2]]) / 255.0
        title = "Unfocused and Reference Images"
    elif len(images) == 2:
        preview = cv2.hconcat([images[0], images[1]]) / 255.0
        title = "Unfocused Images"
    else:
        print("Select at least two images.")
        return

    cv2.imshow(title, preview)
    cv2.setMouseCallback(title, mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def enable_btn():
    """Enable or disable GUI buttons based on how many images are loaded."""
    global images
    count = len(images) if images else 0
    all_buttons = [
        display_selected_images_btn,
        display_calculate_metrics_btn,
        display_decomposition_btn,
        display_fusion_btn,
        display_fusion_decomposition_btn,
    ]

    if count < 2 or count > 3:
        for btn in all_buttons:
            btn.config(state="disabled")
        display_error_btn.config(state="disabled")
    elif count == 2:
        for btn in all_buttons:
            btn.config(state="normal")
        display_error_btn.config(state="disabled")
    else:
        for btn in all_buttons:
            btn.config(state="normal")
        display_error_btn.config(state="normal")


def update_decomposition_label(_event=None):
    decomposition_level_label.config(text=str(int(decomposition_level_scale.get())))


def enable_kernel_selection(_event):
    """Enable wavelet kernel selection only when DWT fusion is chosen."""
    if fusion_model_combo.get() == "DWT":
        dwt_selected_combo.config(state="readonly")
    else:
        dwt_selected_combo.config(state="disabled")


def insert_data_to_metric_table(info):
    """Populate the metric Treeview when metrics were calculated."""
    if info is None:
        return

    for item in metric_table_treeview.get_children():
        metric_table_treeview.delete(item)

    for idx, (key, value) in enumerate(info.items()):
        metric_table_treeview.insert(parent="", index="end", iid=str(idx), values=(key, value))


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Discrete Cosine Transform-based Image Fusion")
    w, h = 520, 440
    window.minsize(w, h)
    window.maxsize(w, h)
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (w / 2))
    y = int((screen_height / 2) - (h / 2))
    window.geometry(f"{w}x{h}+{x}+{y}")

    frame0 = tkinter.ttk.Frame(window)
    frame0.grid(row=0, column=0, padx=10, pady=(30, 15), sticky="w")

    frame1 = tkinter.ttk.Frame(window)
    frame1.grid(row=1, column=0, padx=10, pady=15, sticky="w")

    frame2 = tkinter.ttk.Frame(window)
    frame2.grid(row=2, column=0, padx=10, pady=15, sticky="w")

    frame3 = tkinter.ttk.Frame(window)
    frame3.grid(row=3, column=0, padx=10)

    tkinter.ttk.Button(frame0, text="Select Images\n From Folder",
                       command=select_images_from_folder).grid(row=0, column=0, padx=7)

    tkinter.ttk.Label(frame0, text="Fusion Model: ").grid(row=0, column=1, padx=1)
    fusion_values = ["DCT", "DWT"]
    fusion_model_combo = tkinter.ttk.Combobox(frame0, values=fusion_values, width=6,
                                              justify="left", state="readonly")
    fusion_model_combo.grid(row=0, column=2, padx=7)
    fusion_model_combo.set(fusion_values[0])
    fusion_model_combo.bind("<<ComboboxSelected>>", enable_kernel_selection)

    tkinter.ttk.Label(frame0, text="Decomposition Level: ").grid(row=0, column=3, padx=1)
    decomposition_level_scale = tkinter.ttk.Scale(frame0, from_=1, to=9, orient="horizontal",
                                                  command=update_decomposition_label)
    decomposition_level_scale.grid(row=0, column=4)
    decomposition_level_label = tkinter.ttk.Label(frame0, text="1")
    decomposition_level_label.grid(row=0, column=5)
    decomposition_level_scale.set(1)

    display_selected_images_btn = tkinter.ttk.Button(
        frame1, text="Display\nSelected Images", state="disabled", command=display_selected_images)
    display_selected_images_btn.grid(row=0, column=0, padx=7)

    tkinter.ttk.Label(frame1, text="Kernel: ").grid(row=0, column=1, padx=1)
    dwt_selected_values = ["db1","db2","sym4"]
    dwt_selected_combo = tkinter.ttk.Combobox(frame1, values=dwt_selected_values, width=6,
                                                justify="left", state="disabled")
    dwt_selected_combo.grid(row=0, column=2, padx=7)
    dwt_selected_combo.set(dwt_selected_values[0])

    tkinter.ttk.Label(frame1, text="Details: ").grid(row=0, column=3, padx=1)
    details_values = ["max", "min", "mean"]
    details_combo = tkinter.ttk.Combobox(frame1, values=details_values, width=5, state="readonly")
    details_combo.grid(row=0, column=4, padx=7)
    details_combo.set(details_values[0])

    tkinter.ttk.Label(frame1, text="Approximation: ").grid(row=0, column=5, padx=1)
    approximation_values = ["mean", "min", "max"]
    approximation_combo = tkinter.ttk.Combobox(frame1, values=approximation_values, width=5, state="readonly")
    approximation_combo.grid(row=0, column=6, padx=7)
    approximation_combo.set(approximation_values[0])

    display_decomposition_btn = tkinter.ttk.Button(
        frame2, text="Display\nDecomposition", state="disabled",
        command=lambda: start_fusion(display_decomposition_mosaic=True))
    display_decomposition_btn.grid(row=0, column=0, padx=7)

    display_fusion_decomposition_btn = tkinter.ttk.Button(
        frame2, text="Display\nFusion Decomposition", state="disabled",
        command=lambda: start_fusion(display_fusion_mosaic=True))
    display_fusion_decomposition_btn.grid(row=0, column=1, padx=7)

    display_fusion_btn = tkinter.ttk.Button(
        frame2, text="Display\nFusion", state="disabled",
        command=lambda: start_fusion(display_fusion=True))
    display_fusion_btn.grid(row=0, column=2, padx=7)

    display_error_btn = tkinter.ttk.Button(
        frame2, text="Display\nError", state="disabled",
        command=lambda: start_fusion(display_error=True))
    display_error_btn.grid(row=0, column=3, padx=7)

    tkinter.ttk.Label(frame3, text="Metric Table").grid(row=0, column=1, pady=(0, 7))

    display_calculate_metrics_btn = tkinter.ttk.Button(
        frame3, text="Calculate\n Metrics", state="disabled",
        command=lambda: start_fusion(calculate_metrics=True))
    display_calculate_metrics_btn.grid(row=1, column=0, padx=1)

    metric_table_treeview = tkinter.ttk.Treeview(frame3, height=7)
    metric_table_treeview.grid(row=1, column=1)
    metric_table_treeview["columns"] = ("Metric-Name", "Metric-Value")
    metric_table_treeview.column("#0", width=0, stretch=False)
    metric_table_treeview.column("Metric-Name", width=150, minwidth=100, anchor="center")
    metric_table_treeview.column("Metric-Value", width=150, minwidth=100, anchor="center")
    metric_table_treeview.heading("#0", text="", anchor="w")
    metric_table_treeview.heading("Metric-Name", text="Metric-Name", anchor="center")
    metric_table_treeview.heading("Metric-Value", text="Metric-Value", anchor="center")

    window.mainloop()
