#!/usr/bin/env python3
# =============================================================================
#  lattice_twist_analysis.py
#
#  Lattice parameters and single-crystal lattice twist from synchrotron X-ray
#  nano-diffraction raster scans recorded at a series of sample rotation
#  angles. Developed for (hexagonal) protein crystals in demosponge spicules;
#  applicable to any elongated crystal scanned along its length.
#
#  For each species / dataset the script produces one supplementary figure:
#    (a, b)  hexagonal lattice parameters a and c along the spicule
#    (c-e)   real-space maps of the (100) intensity at three rotation angles
#    (f)     (100) intensity vs rotation angle for each scanned height, with a
#            Gaussian (+ constant background) fit per height
#    (g)     fitted peak angle vs position along the spicule; the slope of a
#            straight-line fit is the twist rate (deg/um)
#  plus CSV files of every plotted series, a fit-summary JSON and a draft
#  figure caption.
#
#  Method summary (details in README.md)
#    Lattice: per scan position, intensity-weighted least squares of 1/d^2
#      against the indexed hkl reflections (all rotation angles pooled);
#      positions are screened for reliability; panels a/b show the mean per
#      height, the band the median +/- robust SD of all reliable positions.
#    Twist: for every scanned height, the (100) intensity (summed across the
#      spicule) vs rotation angle is fitted with a Gaussian; its centre is the
#      peak angle of that height. Heights without a real peak inside the
#      angular scan are excluded. phi_peak = twist * z + const is fitted
#      directly (no inversion); by default unweighted, with the twist
#      uncertainty taken from the fit residuals (PEAK_ERR_MODE).
#
#  Usage
#    1. Set DATA_ROOT and HEATMAP_ROOT below (or the environment variables
#       LATTICE_DATA_ROOT / LATTICE_HEATMAP_ROOT).
#    2. Choose ACTIVE_TAG (one dataset, or "ALL").
#    3. python lattice_twist_analysis.py
# =============================================================================

import os
import re
import csv
import json
import warnings
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning

# =============================================================================
# DATA LOCATIONS  — edit these two lines (or set the environment variables)
# =============================================================================
# DATA_ROOT    : folder that contains one sub-folder per dataset
#                (e.g. TAEshelby/<angle>/<scan>_<frame>/..._peaks_lattice.csv)
# HEATMAP_ROOT : folder under which the (100) heat-map TIFF stacks are stored
DATA_ROOT    = os.environ.get("LATTICE_DATA_ROOT", os.path.join("data", "raw"))
HEATMAP_ROOT = os.environ.get("LATTICE_HEATMAP_ROOT", os.path.join("data", "heatmaps"))

# =============================================================================
# CONFIG  — one block per dataset
# =============================================================================
SAMPLES = {
    "TA": {   # data folder: TAEshelby
        "root":          os.path.join(DATA_ROOT, "TAEshelby"),
        "species":       "Tethya aurantium",
        "prefix_re":     r"^TAeshelby_{angle}_(\d+)_data_000001_\d+$",
        "prefix_fmt":    "TAeshelby_{angle}_{mid}_data_000001_",
        "angles":        list(range(0, 61)),
        "angle_offset":  0,       # raw angle already in tenths of a degree
        "first_frame":   0,
        "last_frame":    98,
        "step_um":       0.5,
        "n_cols":        9,
        "n_rows":        11,
        "row_step_um":   5.0,
        "col_step_um":   0.5,
        "twist_hkl":     (1, 0, 0),
        "twist_fit_range": None,  # unused since v4
        "heatmap_tif":   os.path.join(HEATMAP_ROOT, "TA", "TA Eshelby", "gropu together rotations", "TA 100.tif"),
        "heatmap_z_um":  50.0,
        "heatmap_x_um":  4.0,
        "heatmap_frames": (0, 30, 60),
    },

    "SP": {   # data folder: SPEshelby
        "root":          os.path.join(DATA_ROOT, "SPEshelby"),
        "species":       "Stryphnus ponderosus",
        "prefix_re":     r"^SPeshelby_{angle}_(\d+)_data_000001_\d+$",
        "prefix_fmt":    "SPeshelby_{angle}_{mid}_data_000001_",
        "angles":        list(range(0, 61)),
        "angle_offset":  0,
        "first_frame":   0,
        "last_frame":    98,
        "step_um":       0.5,
        "n_cols":        9,
        "n_rows":        11,
        "row_step_um":   5.0,
        "col_step_um":   0.5,
        "twist_hkl":     (1, 0, 0),
        "twist_fit_range": None,  # unused since v4
        # Looser a and c screening for SP (keys: max_rel, k_sig, k_mad, min_dof)
        "lat_screen":    {"a": {"max_rel": 0.03, "k_sig": 4.0},
                          "c": {"max_rel": 0.03, "k_sig": 4.0}},
        "heatmap_tif":   os.path.join(HEATMAP_ROOT, "SP", "SP Eshelby", "Hitmaps", "SP 100.tif"),
        "heatmap_z_um":  50.0,
        "heatmap_x_um":  4.0,
        "heatmap_frames": (0, 30, 60),
    },

    "RP": {   # data folder: RPEshelby
        "root":          os.path.join(DATA_ROOT, "RPEshelby"),
        "species":       "Rhizaxinella pyrifera",
        "prefix_re":     r"^RPeshelby_{angle}_(\d+)_data_000001_\d+$",
        "prefix_fmt":    "RPeshelby_{angle}_{mid}_data_000001_",
        "angles":        list(range(0, 61)),
        "angle_offset":  0,
        "first_frame":   0,
        "last_frame":    98,
        "step_um":       0.5,
        "n_cols":        9,
        "n_rows":        11,
        "row_step_um":   5.0,
        "col_step_um":   0.5,
        "twist_hkl":     (1, 0, 0),
        "twist_fit_range": None,  # unused since v4
        "heatmap_tif":   os.path.join(HEATMAP_ROOT, "RP", "RP Eshelby", "Hitmaps", "RP 100.tif"),
        "heatmap_z_um":  50.0,
        "heatmap_x_um":  4.0,
        "heatmap_frames": (0, 30, 60),
    },

    "GC": {   # data folder: GCEshelby
        "root":          os.path.join(DATA_ROOT, "GCEshelby"),
        "species":       "Geodia cydonium",
        "prefix_re":     r"^GCeshelby_{angle}_(\d+)_data_000001_\d+$",
        "prefix_fmt":    "GCeshelby_{angle}_{mid}_data_000001_",
        "angles":        list(range(0, 61)),
        "angle_offset":  0,
        "first_frame":   0,
        "last_frame":    98,
        "step_um":       0.5,
        "n_cols":        9,
        "n_rows":        11,
        "row_step_um":   5.0,
        "col_step_um":   0.5,
        "twist_hkl":     (1, 0, 0),
        "twist_fit_range": None,  # unused since v4
        "heatmap_tif":   os.path.join(HEATMAP_ROOT, "GC", "GC", "Hitmaps", "GC 100.tif"),
        "heatmap_z_um":  50.0,
        "heatmap_x_um":  4.0,
        "heatmap_frames": (0, 30, 60),
    },

    "PW": {   # data folder: PWEshelby
        "root":          os.path.join(DATA_ROOT, "PWEshelby"),
        "species":       "Clathria prolifera",
        # Prefix is fully deterministic: mid = angle - 1590, no folder scanning.
        # prefix_re = None signals the fixed-prefix path in detect_prefix().
        "prefix_re":     None,
        "prefix_fmt":    None,
        "angles":        list(range(1590, 1636)),  # 1590-1635 (1636 is empty)
        # Subtract angle_offset before dividing by 10 to get plot degrees:
        # angle 1590 -> 0.0 deg, 1591 -> 0.1 deg, ..., 1635 -> 4.5 deg
        "angle_offset":  1590,
        "first_frame":   0,
        "last_frame":    98,
        "step_um":       0.5,
        "n_cols":        9,
        "n_rows":        11,
        "row_step_um":   5.0,
        "col_step_um":   0.5,
        "twist_hkl":     (1, 0, 0),
        "twist_fit_range": None,  # unused since v4
        # Looser screening for PW (keys: max_rel, k_sig, k_mad, min_dof)
        "lat_screen":    {"a": {"max_rel": 0.04, "k_sig": 4.0},
                          # c relaxed for PW (was max_rel 0.03, k_sig 4.0,
                          # leaving only 9 of 65 positions):
                          "c": {"max_rel": 0.06, "k_sig": 8.0, "k_mad": 4.0,
                                "min_dof": 1}},
        "heatmap_tif":   os.path.join(HEATMAP_ROOT, "PW", "PW Intensity", "Hitmaps 100", "Hitmaps 100.tif"),
        "heatmap_z_um":  50.0,
        "heatmap_x_um":  4.0,
        # TIFF has 32 frames (indices 0-31) = motor angles 1590-1621.
        # Show frames 0, 10, 20 -> motor 1590, 1600, 1610 -> phi = 0, 1, 2 deg relative
        "heatmap_frames": (0, 10, 20),
    },
}

# Which sample to process in this run.
# Options: "TA", "SP", "RP", "GC", "PW",
#          or "ALL" to make the figure for every species in one run
ACTIVE_TAG = "ALL"

# Height axis: True = the first scanned row (frame 0..n_cols-1) is at the far
# end of the spicule (z = 50 um), matching the heatmap TIFFs in panels c-e.
# Can be overridden per sample with "z_flip": False in its SAMPLES block.
Z_FLIP = True

# Output directory for figure + CSVs
OUTPUT_DIR = os.path.join(os.getcwd(), "supplementary_output")

# ---- Lattice-parameter reliability screening (panels a, b) ----------------
LAT_K_MAD        = 3.0    # reject |x - median| > LAT_K_MAD * 1.4826*MAD
LAT_K_SIGMA      = 2.5    # reject sigma > LAT_K_SIGMA * median(sigma)
LAT_MAX_REL_SIG  = 0.015  # reject sigma > this fraction of the value
                          #   (1.5 % -> ~0.9 A for a, ~1.7 A for c); None = off
LAT_MIN_DOF      = 2      # reject fits with fewer than this many degrees of
                          #   freedom (n_reflections - n_params)
LAT_N_ITER       = 3      # iterate median/MAD after each rejection pass
SHOW_REJECTED    = False  # True: draw rejected points as hollow grey markers
LAT_PLOT_MODE    = "mean" # panels a, b: "mean" = one point per height (mean
                          #   of its reliable scan points); "all" = every point
LAT_MEAN_ERR     = "std"  # error bar of the mean: "sem" = SD/sqrt(n), "std" = SD
LAT_X_SPREAD     = 0.6    # "all" mode: the scan points of one height are
                          #   spread over this fraction of the height step,
                          #   ordered by position across the spicule (display only)

# ---- Twist from per-height peak angle (panel g, v4) ------------------------
PEAK_MIN_FRAC    = 0.05   # keep a height only if its peak intensity is at
                          #   least this fraction of the brightest height
PEAK_MIN_SNR     = 5.0    # ... and its peak is >= this many noise SDs above
                          #   that height's median level (rejects noise rows)
PEAK_DROP_EDGES  = True   # drop heights whose Gaussian centre is pinned at the
                          #   first/last scanned angle (tail of a peak outside
                          #   the scan; its angle is meaningless)
PEAK_EDGE_NSIG   = 0.0    # > 0: also drop peaks whose centre lies within this
                          #   many std of the scan edge (stricter; 0 = off, so
                          #   partly cut but well-located peaks are kept)
PEAK_MIN_HEIGHTS = 3      # need at least this many heights to fit a line
PEAK_PROFILE     = "sum"  # per-height histogram: "sum" over x, or "max" pixel
PEAK_WIN_DEG     = 0.8    # Gaussian fit window: +/- this around the highest bin
PEAK_ERR_MODE    = "residual" # how the twist fit treats the per-height errors:
                          #   "gauss" = weighted fit (1/err^2) with each peak's
                          #       Gaussian-fit error of the centre; the twist
                          #       error is propagated from these errors only
                          #       (no scaling). The scatter-based error is
                          #       still reported alongside for reference.
                          #   "auto" (recommended) = weighted fit with the fit
                          #       uncertainty of each peak position; if those
                          #       errors do not explain the scatter (chi^2
                          #       test, see PEAK_CHI2_P), fall back to an
                          #       unweighted fit with the error from residuals
                          #   "residual" = always unweighted, error from residuals
                          #   "sem" = weighted by peak-position uncertainty
                          #       (floored at the angle step / sqrt(12))
                          #   "std" = weighted by the peak width (std);
                          #       conservative, not a position uncertainty
PEAK_SHOW_MIN_SNR = 6.0   # panel f: also draw heights NOT used in the fit
                          #   (weak / cut by scan edge) if a real Gaussian
                          #   fits and S/N >= this (grey, dotted)
SHOW_EXCLUDED    = False  # panels f, g: also draw heights NOT used in the fit
                          #   (grey). False = only heights in the fit are drawn
SHOW_ALL_HEIGHTS = False  # panels f, g: show every height with a fitted peak,
                          #   including those not used in the twist fit
                          #   (grey). False = only those passing PEAK_SHOW_MIN_SNR
G_ERRBAR         = "gauss" # panel g error bars: "gauss" = each height's own
                          #   Gaussian-fit error of the peak centre (from f);
                          #   "scatter" = residual standard error of the line
                          #   (same for all points). The twist error is set by
                          #   PEAK_ERR_MODE either way.
SHOW_RESIDUALS   = False  # panel g: add a residual strip (measured - fitted
                          #   peak angle) below the fit
MOTOR_ERR_DEG    = 0.0    # rotation-motor position uncertainty (deg, 1 sigma),
                          #   added in quadrature to the Gaussian-fit error of
                          #   each peak angle. 0 = unknown: the per-angle error
                          #   is then estimated from the scatter about the fit
                          #   ("auto"). Set it once the beamline value is known.
PEAK_CHI2_P      = 0.01   # "auto": the errors are accepted if the chi^2
                          #   probability lies between PEAK_CHI2_P and
                          #   1 - PEAK_CHI2_P

# =============================================================================
# NATURE-PHYSICS STYLE
# =============================================================================
MM = 1.0 / 25.4
FIG_W_MM = 183.0
FIG_H_MM = 210.0

def set_nature_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":   "dejavusans",
        "font.size":          7,
        "axes.labelsize":     8,
        "axes.titlesize":     8,
        "axes.linewidth":     0.6,
        "axes.labelpad":      2.0,
        "xtick.labelsize":    6.5,
        "ytick.labelsize":    6.5,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.size":   1.3,
        "ytick.minor.size":   1.3,
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,
        "legend.fontsize":    6,
        "legend.frameon":     False,
        "legend.handlelength":  1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing":  0.3,
        "lines.linewidth":    0.9,
        "lines.markersize":   3.0,
        "figure.dpi":         150,
        "savefig.dpi":        600,
        "figure.figsize":     (7.2, 8.3),
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
    })

# Colour-blind-safe palette (Wong / Nature-friendly)
C_DATA  = "#0072B2"   # blue   — per-frame data points
C_GAUS  = "#E69F00"   # orange — global median
C_FIT   = "#CC3311"   # red    — twist linear fit
C_CENT  = "#0072B2"   # blue   — twist data points
C_EXCL  = "0.55"      # grey   — excluded points
# Height colours (panel f): red-free spectrum, violet -> blue -> teal ->
# green -> amber (red is reserved for the twist fit line)
HEIGHT_RAMP = ["#5E4FA2", "#3B6FB6", "#1F8FC0", "#0FA39E",
               "#2EAA62", "#6FA82A", "#A39C14", "#C4800C"]

# =============================================================================
# PREFIX DETECTION
# Standard datasets (TA/SP/RP/GC): scan the angle subfolder for the mid number.
# PW dataset: mid = angle - 1590, prefix is fully deterministic.
# =============================================================================
def detect_prefix(root, cfg, angle):
    """Return the scan prefix string for a given angle, or None if not found."""
    prefix_re  = cfg["prefix_re"]
    prefix_fmt = cfg["prefix_fmt"]
    offset     = cfg.get("angle_offset", 0)

    angle_dir = os.path.join(root, str(angle))
    if not os.path.isdir(angle_dir):
        return None

    # PW: deterministic prefix, no scanning needed
    if prefix_re is None:
        mid = angle - offset
        return f"PWeshelby_{mid}_{angle}_data_000001_"

    # Standard: scan directory for a folder matching the regex
    pat = re.compile(prefix_re.format(angle=angle), re.IGNORECASE)
    for name in os.listdir(angle_dir):
        m = pat.match(name)
        if m:
            return prefix_fmt.format(angle=angle, mid=int(m.group(1)))
    return None


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def row_to_z(row, n_rows, row_step, z_flip):
    """
    Position along the spicule (um) of scan row `row` (frame // n_cols).
    z_flip = True : the first scanned row is at the far end (z_max), as in the
                    heatmap TIFFs (panels c-e), so all panels share one axis.
    z_flip = False: the first scanned row is at z = 0.
    """
    return float((n_rows - 1 - row) * row_step if z_flip else row * row_step)


def parse_peaks_from_csv(csv_path):
    peaks = []
    if not os.path.isfile(csv_path):
        return peaks
    with open(csv_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                peaks.append({
                    "h": int(parts[0]), "k": int(parts[1]), "l": int(parts[2]),
                    "I_max": float(parts[5]), "d_exp": float(parts[6]),
                    "rel":   float(parts[9]),
                })
            except ValueError:
                continue
    return peaks


def weighted_lstsq_hexagonal(peaks):
    """Intensity-weighted 1/d^2 LSQ -> (a, c, sigma_a, sigma_c, dof, wRMS%)."""
    valid = [p for p in peaks
             if np.isfinite(p["d_exp"]) and p["d_exp"] > 0
             and np.isfinite(p["I_max"]) and p["I_max"] > 0]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan
    h_ = np.array([p["h"] for p in valid], float)
    k_ = np.array([p["k"] for p in valid], float)
    l_ = np.array([p["l"] for p in valid], float)
    d_ = np.array([p["d_exp"] for p in valid], float)
    Iw = np.array([p["I_max"] for p in valid], float)

    w     = Iw / Iw.sum()
    y     = 1.0 / d_**2
    A_col = (4.0/3.0) * (h_**2 + h_*k_ + k_**2)
    B_col = l_**2
    has_a = np.any(A_col != 0)
    has_c = np.any(B_col != 0)
    dof   = len(valid) - int(has_a) - int(has_c)

    cols = []
    if has_a: cols.append(A_col)
    if has_c: cols.append(B_col)
    M    = np.column_stack(cols)
    MTWM = M.T @ np.diag(w) @ M
    MTWy = M.T @ np.diag(w) @ y
    try:
        x, *_ = np.linalg.lstsq(MTWM, MTWy, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, dof, np.nan

    i = 0
    u = x[i] if has_a else np.nan
    if has_a: i += 1
    v = x[i] if has_c else np.nan
    if has_a and u <= 0: return np.nan, np.nan, np.nan, np.nan, dof, np.nan
    if has_c and v <= 0: return np.nan, np.nan, np.nan, np.nan, dof, np.nan

    a_ref  = 1.0/np.sqrt(u) if has_a else np.nan
    c_ref  = 1.0/np.sqrt(v) if has_c else np.nan
    y_calc = M @ x
    chi2_r = np.sum(w * (y - y_calc)**2) / max(dof, 1)
    try:
        cov = np.linalg.inv(MTWM) * chi2_r
    except np.linalg.LinAlgError:
        return a_ref, c_ref, np.nan, np.nan, dof, np.nan
    i = 0
    sa = sc = np.nan
    if has_a: sa = np.sqrt(abs(cov[i, i])) / (2.0*u**1.5); i += 1
    if has_c: sc = np.sqrt(abs(cov[i, i])) / (2.0*v**1.5)
    # With dof <= 0 the fit is exactly determined: sigma is meaningless
    if dof <= 0:
        sa = sc = np.nan
    d_calc = 1.0/np.sqrt(np.maximum(y_calc, 1e-30))
    wRMS   = np.sqrt(np.sum(w * ((d_ - d_calc)/d_)**2)) * 100.0
    return a_ref, c_ref, sa, sc, dof, wRMS


def pick_max_I_per_hkl(peak_list):
    best = {}
    for p in peak_list:
        key = (p["h"], p["k"], p["l"])
        if key not in best or p["I_max"] > best[key]["I_max"]:
            best[key] = p
    return list(best.values())


def gaussian_bg(x, A, mu, sigma, bg):
    return A * np.exp(-0.5*((x - mu)/sigma)**2) + bg


def fit_gaussian_profile(frames_arr, I_arr):
    if len(frames_arr) < 3:
        return None
    x = np.asarray(frames_arr, float); y = np.asarray(I_arr, float)
    bg0  = np.percentile(y, 10); A0 = y.max() - bg0
    mu0  = x[np.argmax(y)];      sig0 = max((x.max() - x.min())/4.0, 1.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                gaussian_bg, x, y, p0=[A0, mu0, sig0, bg0],
                bounds=([0, x.min()-1, 0.3, -np.inf],
                        [np.inf, x.max()+1, x.max()-x.min()+2, np.inf]),
                maxfev=5000)
        A, mu, sig, bg = popt
        if A <= 0 or not (x.min()-1 <= mu <= x.max()+1) or sig < 0.3:
            return None
        return A, mu, sig, bg
    except Exception:
        return None


def pick_gaussian_peak_per_hkl(angle_frame_peaks, angles, f0, f1):
    hkl_angle = defaultdict(lambda: defaultdict(list))
    for a in angles:
        for fr in range(f0, f1+1):
            for p in angle_frame_peaks[a][fr]:
                hkl_angle[(p["h"], p["k"], p["l"])][a].append(
                    (fr, p["I_max"], p["d_exp"]))
    best_per_hkl = {}
    for key, ad in hkl_angle.items():
        best_A, best = -np.inf, None
        for a, obs in ad.items():
            fr = [o[0] for o in obs]; I = [o[1] for o in obs]; d = [o[2] for o in obs]
            res = fit_gaussian_profile(fr, I)
            if res:
                A, mu, sig, bg = res
                order   = np.argsort(fr)
                d_at_mu = float(np.interp(mu, np.asarray(fr)[order],
                                          np.asarray(d)[order]))
                cand = {"h": key[0], "k": key[1], "l": key[2],
                        "I_max": A, "d_exp": d_at_mu}
            else:
                j    = int(np.argmax(I))
                cand = {"h": key[0], "k": key[1], "l": key[2],
                        "I_max": I[j], "d_exp": d[j]}
            if cand["I_max"] > best_A:
                best_A, best = cand["I_max"], cand
        if best:
            best_per_hkl[key] = best
    return list(best_per_hkl.values())


def weighted_centroid_2d(I_2d, row_um, col_um):
    total = I_2d.sum()
    if total <= 0:
        return np.nan, np.nan
    z = np.sum(row_um[:, None]*I_2d) / total
    x = np.sum(col_um[None, :]*I_2d) / total
    return z, x


def peak_position_2d(I_2d, row_um, col_um):
    if I_2d.max() <= 0:
        return np.nan, np.nan
    idx = np.unravel_index(np.argmax(I_2d), I_2d.shape)
    return row_um[idx[0]], col_um[idx[1]]


# =============================================================================
# ROBUST LATTICE STATISTICS
# =============================================================================
def robust_reliable_mask(vals, errs, dof=None, k_mad=LAT_K_MAD,
                         k_sig=LAT_K_SIGMA, max_rel=LAT_MAX_REL_SIG,
                         min_dof=LAT_MIN_DOF, n_iter=LAT_N_ITER):
    """
    Flag per-position lattice values as reliable.
    A point is rejected if
      - value or sigma is not finite (under-determined fit, dof <= 0), or
      - the fit has fewer than min_dof degrees of freedom, or
      - sigma > max_rel * value (absolute precision cap), or
      - sigma > k_sig * median(sigma of currently reliable points), or
      - |value - median| > k_mad * 1.4826 * MAD  (MAD of reliable points).
    Median / MAD / median-sigma are re-estimated each iteration.
    """
    vals = np.asarray(vals, float); errs = np.asarray(errs, float)
    keep = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
    if dof is not None and min_dof:
        keep &= np.asarray(dof) >= min_dof
    if max_rel is not None:
        keep &= errs <= max_rel * np.abs(vals)
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        med     = np.median(vals[keep])
        mad     = 1.4826 * np.median(np.abs(vals[keep] - med))
        sig_med = np.median(errs[keep])
        new = keep.copy()
        if mad > 0:
            new &= np.abs(vals - med) <= k_mad * mad
        if sig_med > 0:
            new &= errs <= k_sig * sig_med
        if np.array_equal(new, keep):
            break
        keep = new
    return keep


def screen_reasons(vals, errs, dof, ok, k_mad=LAT_K_MAD, k_sig=LAT_K_SIGMA,
                   max_rel=LAT_MAX_REL_SIG, min_dof=LAT_MIN_DOF, n_iter=None):
    """Short summary of why per-position fits were rejected (first failing rule)."""
    vals = np.asarray(vals, float); errs = np.asarray(errs, float)
    dof = np.asarray(dof)
    fin = np.isfinite(vals)
    cnt = {"no sigma": 0, f"dof<{min_dof}": 0, f"sigma>{max_rel:g}*value": 0,
           "sigma/value outlier": 0}
    for i in np.where(fin & ~ok)[0]:
        if not (np.isfinite(errs[i]) and errs[i] > 0):
            cnt["no sigma"] += 1
        elif min_dof and dof[i] < min_dof:
            cnt[f"dof<{min_dof}"] += 1
        elif max_rel is not None and errs[i] > max_rel * abs(vals[i]):
            cnt[f"sigma>{max_rel:g}*value"] += 1
        else:
            cnt["sigma/value outlier"] += 1
    return ", ".join(f"{k} {v}" for k, v in cnt.items() if v) or "none"


def robust_summary(vals, mask):
    """Median, robust SD (1.4826*MAD) and std. error of the median."""
    v = np.asarray(vals, float)[mask]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    med = float(np.median(v))
    rsd = float(1.4826 * np.median(np.abs(v - med)))
    sem = float(1.2533 * rsd / np.sqrt(v.size))
    return med, rsd, sem, int(v.size)


# =============================================================================
# TWIST FROM PER-HEIGHT PEAK ANGLE  (v4)
# =============================================================================
def _peak_stats_1d(phi, y, win_deg):
    """
    Peak location and standard deviation of one intensity-vs-angle histogram.

    1. Gaussian + constant background fitted in a window of +/- win_deg
       around the highest bin:  mu = peak location, sigma = peak std,
       mu_err = 1-sigma fit uncertainty of mu.
    2. If the fit fails or is unphysical, fall back to intensity-weighted
       moments (background-subtracted) in the same window.
    Returns dict(mu, sigma, mu_err, A, bg, method).
    """
    dphi = float(np.median(np.diff(phi))) if len(phi) > 1 else 0.1
    k    = int(np.argmax(y))
    sel  = np.abs(phi - phi[k]) <= win_deg + 1e-9
    x, v = phi[sel], y[sel]

    bg0 = float(np.percentile(y, 20))
    A0  = float(y[k] - bg0)
    if len(x) >= 5 and A0 > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, cov = curve_fit(
                    gaussian_bg, x, v, p0=[A0, phi[k], 2*dphi, bg0],
                    bounds=([0, x.min(), dphi/3, -np.inf],
                            [np.inf, x.max(), win_deg, np.inf]),
                    maxfev=10000)
            A, mu, sig, bg = p
            perr = np.sqrt(np.abs(np.diag(cov)))
            if (A > 0 and np.all(np.isfinite(perr))
                    and x.min() <= mu <= x.max()):
                return dict(mu=float(mu), sigma=float(abs(sig)),
                            mu_err=float(perr[1]), A=float(A), bg=float(bg),
                            method="gauss")
        except Exception:
            pass

    # moment fallback
    w = np.clip(v - np.min(v), 0, None)
    if w.sum() <= 0:
        return dict(mu=float(phi[k]), sigma=dphi/np.sqrt(12),
                    mu_err=dphi/np.sqrt(12), A=A0, bg=bg0, method="argmax")
    mu  = float(np.sum(w*x) / w.sum())
    sig = float(np.sqrt(max(np.sum(w*(x-mu)**2) / w.sum(), dphi**2/12)))
    neff = w.sum()**2 / np.sum(w**2)
    return dict(mu=mu, sigma=sig, mu_err=sig/np.sqrt(max(neff, 1.0)),
                A=float(w.max()), bg=float(np.min(v)), method="moments")


def peak_stats_per_height(I_cube, angle_degs, row_um,
                          profile=PEAK_PROFILE, win_deg=PEAK_WIN_DEG,
                          err_mode=PEAK_ERR_MODE, min_snr=PEAK_MIN_SNR,
                          edge_nsig=PEAK_EDGE_NSIG,
                          min_frac=PEAK_MIN_FRAC, drop_edges=PEAK_DROP_EDGES):
    """
    One point per scanned height.

    I_cube : (n_angles, n_rows, n_cols) (hkl) intensity per pixel.
    For each height (row):
      1. intensity histogram vs rotation angle: sum over x ("sum") or the
         brightest pixel across x ("max")
      2. peak location mu and standard deviation sigma from a Gaussian fit
         (see _peak_stats_1d)
      3. error used in the twist fit: sigma ("std") or the fit uncertainty
         of mu ("sem"); never below the angle-step quantisation dphi/sqrt(12)
    A height is used in the fit only if it has real signal (peak >= min_frac
    of the brightest height) and, if drop_edges, its highest bin is not on the
    first or last scanned angle.
    """
    angle_degs = np.asarray(angle_degs, float)
    n_rows = len(row_um)
    dphi   = float(np.median(np.diff(angle_degs))) if len(angle_degs) > 1 else 0.1
    H      = I_cube.sum(axis=2) if profile == "sum" else I_cube.max(axis=2)

    mu = np.full(n_rows, np.nan); sig = np.full(n_rows, np.nan)
    mu_err = np.full(n_rows, np.nan); A = np.zeros(n_rows); bg = np.zeros(n_rows)
    method = [""] * n_rows
    I_peak = H.max(axis=0); k_peak = H.argmax(axis=0)
    for r in range(n_rows):
        if I_peak[r] <= 0:
            continue
        s = _peak_stats_1d(angle_degs, H[:, r], win_deg)
        mu[r], sig[r], mu_err[r] = s["mu"], s["sigma"], s["mu_err"]
        A[r], bg[r], method[r]   = s["A"], s["bg"], s["method"]

    # error bar on each peak angle
    q_floor = dphi / np.sqrt(12)
    if err_mode == "std":
        err = np.maximum(sig, q_floor)
    elif err_mode == "sem":
        err = np.sqrt(np.maximum(mu_err, q_floor)**2 + MOTOR_ERR_DEG**2)
    elif err_mode == "gauss":
        err = mu_err.copy()                 # Gaussian-fit error of the centre only
    else:                                   # "auto" / "residual"
        # Gaussian-fit error of the centre + motor uncertainty, in quadrature
        err = np.sqrt(mu_err**2 + MOTOR_ERR_DEG**2)
    err = np.where(np.isfinite(err) & (err > 0), err, np.nan)

    # signal-to-noise of each height: peak above the median level, in units
    # of the robust noise (1.4826*MAD) of that height's histogram
    med_r = np.median(H, axis=0)
    noise = 1.4826 * np.median(np.abs(H - med_r), axis=0)
    noise = np.where(noise > 0, noise, np.sqrt(np.maximum(med_r, 1.0)))
    snr   = (I_peak - med_r) / noise

    ref  = I_peak.max() if I_peak.size else 0.0
    used = np.ones(n_rows, bool)
    note = [""] * n_rows
    for r in range(n_rows):
        if I_peak[r] <= 0:
            used[r] = False; note[r] = "no signal"
        elif snr[r] < min_snr:
            used[r] = False; note[r] = f"noise (S/N {snr[r]:.1f})"
        elif I_peak[r] < min_frac * ref:
            used[r] = False; note[r] = "weak"
        elif not np.isfinite(sig[r]) or sig[r] < 0.5*dphi:
            used[r] = False; note[r] = "unresolved spike (width < half a step)"
        elif drop_edges and (
                abs(mu[r] - angle_degs.min()) < 0.1*dphi
                or abs(mu[r] - angle_degs.max()) < 0.1*dphi):
            used[r] = False; note[r] = "fit pinned at scan edge"
        elif drop_edges and edge_nsig > 0 and (
                mu[r] - edge_nsig*sig[r] < angle_degs.min()
                or mu[r] + edge_nsig*sig[r] > angle_degs.max()):
            used[r] = False; note[r] = "peak cut by scan edge"
        elif method[r] != "gauss":
            note[r] = f"{method[r]} estimate"
    return {"z_um": np.asarray(row_um, float), "snr": snr,
            "phi_peak_deg": mu,
            "phi_std_deg": sig, "phi_mu_err_deg": mu_err, "phi_err_deg": err,
            "A": A, "bg": bg, "method": method, "I_peak": I_peak,
            "used": used, "note": note, "I_row": H, "profile": profile,
            "err_mode": err_mode}


def show_in_f(PH, dphi, phi_lo, phi_hi, min_snr=PEAK_SHOW_MIN_SNR):
    """
    Heights NOT used in the twist fit that are still drawn (grey) in panels
    f and g: a real signal (S/N >= min_snr) with a Gaussian wider than a
    single-bin spike. This includes weak tails whose centre sits at the edge
    of the angular scan (their angle is a lower/upper bound, not used in the
    fit); pure-noise heights are not drawn.
    """
    n = len(PH["z_um"])
    out = np.zeros(n, bool)
    if not SHOW_EXCLUDED:
        return out                      # only heights used in the fit are drawn
    for r in range(n):
        if PH["used"][r]:
            continue
        mu, sd = PH["phi_peak_deg"][r], PH["phi_std_deg"][r]
        out[r] = (np.isfinite(mu) and np.isfinite(sd)
                  and sd > 0.5 * dphi
                  and PH["snr"][r] >= min_snr)
    return out


def _wls(x, y, w):
    """Weighted LSQ line; returns beta, (X^T W X)^-1, residuals."""
    X = np.column_stack([x, np.ones_like(x)])
    C = np.linalg.inv(X.T @ (w[:, None] * X))
    b = C @ X.T @ (w * y)
    return b, C, y - X @ b


def fit_twist(z, phi, err, used, mode=PEAK_ERR_MODE, p_lim=PEAK_CHI2_P,
              min_pts=PEAK_MIN_HEIGHTS):
    """
    Straight line  phi = slope*z + intercept;  slope = twist (deg/um).

    1. Weighted fit, w = 1/err^2. The per-height errors are propagated
       into the slope through the covariance (X^T W X)^-1.
    2. Goodness of fit: chi^2 with n-2 degrees of freedom and its
       probability P(chi^2 >= observed).
    3. Which error is reported:
       "auto"     : if p_lim < P < 1-p_lim the errors describe the scatter
                    -> weighted fit, propagated error (scaled by
                    sqrt(chi^2_red) if > 1). Otherwise the errors are not
                    trustworthy -> unweighted fit, error from the residuals.
       "residual" : always unweighted, error from the residuals.
       "sem"/"std": weighted fit, error scaled by sqrt(chi^2_red) if > 1.
    The unweighted fit with residual error is the ordinary least-squares
    standard error of the slope (as scipy.stats.linregress).
    """
    m = used & np.isfinite(phi)
    if mode != "residual":
        m &= np.isfinite(err) & (err > 0)
    n = int(m.sum())
    if n < min_pts:
        return None
    x, y = z[m], phi[m]
    dof = n - 2

    # weighted fit with the given errors (for propagation / chi^2 test)
    chi2 = chi2_red = p_chi2 = np.nan
    if mode != "residual":
        w = 1.0 / err[m]**2
        b_w, C_w, r_w = _wls(x, y, w)
        chi2     = float(np.sum(w * r_w**2))
        chi2_red = chi2 / dof if dof > 0 else np.nan
        p_chi2   = float(stats.chi2.sf(chi2, dof)) if dof > 0 else np.nan

    use_weighted = (mode in ("sem", "std", "gauss")) or (
        mode == "auto" and np.isfinite(p_chi2) and p_lim < p_chi2 < 1 - p_lim)

    if use_weighted:
        beta, cov_prop = b_w, C_w
        if mode == "gauss":
            cov = cov_prop          # pure propagation of the Gaussian-fit errors
        else:
            cov = cov_prop * (max(1.0, chi2_red) if np.isfinite(chi2_red) else 1.0)
        res = r_w
        weighting = "weighted (1/err^2), errors propagated"
        ws = w
    else:
        beta, C1, res = _wls(x, y, np.ones_like(x))
        s2  = float(np.sum(res**2) / dof) if dof > 0 else np.nan
        cov = C1 * s2
        cov_prop = C_w if mode != "residual" else cov
        weighting = "unweighted, error from residuals"
        ws = np.ones_like(x)

    # scatter-based (ordinary least-squares) error, always kept for reference
    _, C1r, res_r = _wls(x, y, np.ones_like(x))
    stderr_scatter = (float(np.sqrt(C1r[0, 0] * np.sum(res_r**2) / dof))
                      if dof > 0 else np.nan)
    ybar   = np.sum(ws*y) / np.sum(ws)
    ss_tot = np.sum(ws*(y - ybar)**2)
    r2     = 1.0 - np.sum(ws*res**2)/ss_tot if ss_tot > 0 else np.nan
    return {"slope": float(beta[0]), "intercept": float(beta[1]),
            "stderr": float(np.sqrt(cov[0, 0])),
            "stderr_propagated": float(np.sqrt(cov_prop[0, 0])),
            "stderr_scatter": stderr_scatter,
            "intercept_err": float(np.sqrt(cov[1, 1])),
            "cov": cov, "chi2": chi2, "dof": dof, "chi2_red": chi2_red,
            "p_chi2": p_chi2, "weighting": weighting,
            "weighted": bool(use_weighted),
            "resid_rms_deg": float(np.sqrt(np.sum(res**2)/dof)) if dof > 0 else np.nan,
            "r2": float(r2), "n": n,
            "z_min": float(x.min()), "z_max": float(x.max())}


def fit_band(fs, z):
    """1-sigma confidence band of the fitted phi(z) from the covariance."""
    J = np.column_stack([z, np.ones_like(z)])
    return np.sqrt(np.einsum("ij,jk,ik->i", J, fs["cov"], J))


def fit_line(phi, y, mask):
    """Old centroid fit (z vs phi). Kept only for comparison in the JSON."""
    m = mask & np.isfinite(y)
    if m.sum() < 6:
        return None
    r = stats.linregress(phi[m], y[m])
    return {"slope": r.slope, "intercept": r.intercept,
            "r2": r.rvalue**2, "stderr": r.stderr, "n": int(m.sum()),
            "phi_min": float(phi[m].min()), "phi_max": float(phi[m].max())}


# =============================================================================
# CSV EXPORT HELPER
# =============================================================================
def export_csv(path, header, columns):
    n = max(len(c) for c in columns)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for i in range(n):
            row = []
            for c in columns:
                row.append("" if i >= len(c) else
                           ("" if c[i] is None or
                            (isinstance(c[i], float) and np.isnan(c[i]))
                            else c[i]))
            wr.writerow(row)
    print(f"  CSV : {path}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis(cfg):
    root       = cfg["root"]
    angles     = cfg["angles"]
    f0, f1     = cfg["first_frame"], cfg["last_frame"]
    step       = cfg["step_um"]
    n_cols     = cfg["n_cols"]
    n_rows     = cfg["n_rows"]
    row_step   = cfg.get("row_step_um", step)
    col_step   = cfg.get("col_step_um", step)
    twist_hkl  = cfg["twist_hkl"]
    offset     = cfg.get("angle_offset", 0)
    z_flip     = cfg.get("z_flip", Z_FLIP)

    # Build prefix cache
    prefix_cache = {}
    for a in angles:
        pf = detect_prefix(root, cfg, a)
        if pf:
            prefix_cache[a] = pf
    if not prefix_cache:
        raise RuntimeError(f"No angle subfolders found under {root}")

    # Collect peaks
    angle_frame_peaks = defaultdict(lambda: defaultdict(list))
    master     = []
    peak_grid  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for a in angles:
        pf = prefix_cache.get(a)
        if pf is None:
            continue
        adir = os.path.join(root, str(a))
        for fr in range(f0, f1+1):
            folder = os.path.join(adir, pf + f"{fr:06d}")
            csvf   = os.path.join(folder, pf + f"{fr:06d}_peaks_lattice.csv")
            if not os.path.isdir(folder):
                continue
            for p in parse_peaks_from_csv(csvf):
                pt = {**p, "angle": a, "frame": fr}
                angle_frame_peaks[a][fr].append(pt)
                master.append(pt)
                row, col = divmod(fr, n_cols)
                peak_grid[a][row][col].append(pt)

    if not master:
        raise RuntimeError("No peaks parsed. Check paths / prefix pattern.")

    # Global lattice parameters (kept for comparison / CSV / JSON)
    raw_peaks                         = pick_max_I_per_hkl(master)
    a_raw, c_raw, sa_raw, sc_raw, _, _ = weighted_lstsq_hexagonal(raw_peaks)
    gaus_peaks                        = pick_gaussian_peak_per_hkl(
                                            angle_frame_peaks, angles, f0, f1)
    a_gaus, c_gaus, sa_gaus, sc_gaus, _, _ = weighted_lstsq_hexagonal(gaus_peaks)

    # Per-frame lattice parameters
    frames_s, heights, a_pf, sa_pf, c_pf, sc_pf, dof_pf = [], [], [], [], [], [], []
    x_pf = []
    for fr in range(f0, f1+1):
        allf = []
        for a in angles:
            allf.extend(angle_frame_peaks[a][fr])
        if not allf:
            continue
        fmI = pick_max_I_per_hkl(allf)
        av, cv, sav, scv, dv, _ = weighted_lstsq_hexagonal(fmI)
        # Raster: frames run across the spicule first (n_cols points,
        # col_step apart), then step along the spicule (row_step).
        row_f, col_f = divmod(fr, n_cols)
        frames_s.append(fr); dof_pf.append(dv)
        heights.append(row_to_z(row_f, n_rows, row_step, z_flip))
        x_pf.append(col_f * col_step)
        a_pf.append(av);  sa_pf.append(sav)
        c_pf.append(cv);  sc_pf.append(scv)
    heights = np.array(heights); x_pf = np.array(x_pf)
    frames_s = np.array(frames_s)
    a_pf    = np.array(a_pf);  sa_pf = np.array(sa_pf)
    c_pf    = np.array(c_pf);  sc_pf = np.array(sc_pf)
    dof_pf  = np.array(dof_pf)

    # Reliability screening + robust global values
    scr  = cfg.get("lat_screen", {})
    a_ok = robust_reliable_mask(a_pf, sa_pf, dof_pf, **scr.get("a", {}))
    c_ok = robust_reliable_mask(c_pf, sc_pf, dof_pf, **scr.get("c", {}))
    a_med, a_rsd, a_sem, a_n = robust_summary(a_pf, a_ok)
    c_med, c_rsd, c_sem, c_n = robust_summary(c_pf, c_ok)
    print(f"  a: {a_n}/{np.isfinite(a_pf).sum()} reliable  "
          f"median = {a_med:.2f} +/- {a_rsd:.2f} A (robust SD)")
    print(f"  c: {c_n}/{np.isfinite(c_pf).sum()} reliable  "
          f"median = {c_med:.2f} +/- {c_rsd:.2f} A (robust SD)")
    for nm, v, e, ok, key in (("a", a_pf, sa_pf, a_ok, "a"), ("c", c_pf, sc_pf, c_ok, "c")):
        print(f"     {nm} rejections: " + screen_reasons(v, e, dof_pf, ok,
                                                        **scr.get(key, {})))

    # Twist analysis: build intensity cube for tracked hkl
    # row_um[r] = position along the spicule of scan row r
    row_um     = np.array([row_to_z(r, n_rows, row_step, z_flip)
                           for r in range(n_rows)])
    col_um     = np.arange(n_cols) * col_step
    # angle_degs: physical rotation degrees (PW: 0..4.5, others: 0..6)
    angle_degs = np.array([(a - offset) / 10.0 for a in angles])
    I_cube     = np.zeros((len(angles), n_rows, n_cols))
    for ai, a in enumerate(angles):
        for r in range(n_rows):
            for c in range(n_cols):
                for p in peak_grid[a][r][c]:
                    if (p["h"], p["k"], p["l"]) == twist_hkl:
                        if p["I_max"] > I_cube[ai, r, c]:
                            I_cube[ai, r, c] = p["I_max"]

    # Old per-angle quantities: kept for panel f and the CSV only
    z_cent = np.full(len(angles), np.nan)
    z_max  = np.full(len(angles), np.nan)
    I_tot  = np.zeros(len(angles))
    for ai in range(len(angles)):
        sl        = I_cube[ai]
        I_tot[ai] = sl.sum()
        if sl.max() > 0:
            z_cent[ai], _ = weighted_centroid_2d(sl, row_um, col_um)
            z_max[ai],  _ = peak_position_2d(sl, row_um, col_um)

    # ---- v5 twist: per-height histogram -> peak +/- std -> weighted fit ----
    PH = peak_stats_per_height(I_cube, angle_degs, row_um)
    fs_ph = fit_twist(PH["z_um"], PH["phi_peak_deg"],
                      PH["phi_err_deg"], PH["used"])
    fs_old = fit_line(angle_degs, z_cent, np.isfinite(z_cent))

    print(f"  per-height peaks ({PH['profile']} over x, error = {PH['err_mode']}"
          + (f", motor error {MOTOR_ERR_DEG:g} deg added in quadrature"
             if MOTOR_ERR_DEG > 0 else "") + "):")
    for r in range(len(PH["z_um"])):
        u = PH["used"][r]
        print(f"    z = {PH['z_um'][r]:5.1f} um  "
              f"peak = {PH['phi_peak_deg'][r]:5.2f} deg  "
              f"std = {PH['phi_std_deg'][r]:5.3f}  "
              f"pos.err = {PH['phi_mu_err_deg'][r]:6.4f}  "
              f"I = {PH['I_peak'][r]:9.1f}  [{PH['method'][r]}]  "
              f"{'USED' if u else 'skip (' + PH['note'][r] + ')'}")
    if fs_ph is not None:
        print(f"  chi^2 test of the per-height errors: chi2_red = "
              f"{fs_ph['chi2_red']:.2f} (dof {fs_ph['dof']}), "
              f"P = {fs_ph['p_chi2']:.3g}")
        print(f"  fit used: {fs_ph['weighting']}")
        print(f"  twist = {fs_ph['slope']:+.4f} +/- {fs_ph['stderr']:.4f} "
              f"deg/um  (1 sigma, {fs_ph['n']} heights, residual rms "
              f"{fs_ph['resid_rms_deg']:.3f} deg, R^2 = {fs_ph['r2']:.4f})")
    else:
        print(f"  WARNING: fewer than {PEAK_MIN_HEIGHTS} usable heights, "
              f"no twist fit")
    if fs_old is not None:
        print(f"  (old centroid method would give "
              f"{1/abs(fs_old['slope']):.4f} deg/um)")

    # Load heatmap TIFF (optional)
    heatmaps = None
    tif_path = cfg.get("heatmap_tif", "")
    if tif_path:
        if not os.path.isfile(tif_path):
            base = tif_path.replace("\\", "/").split("/")[-1]
            alt  = os.path.join(os.getcwd(), base)
            if os.path.isfile(alt):
                tif_path = alt
        if os.path.isfile(tif_path):
            from PIL import Image as _Image
            stack = _Image.open(tif_path)
            want  = cfg.get("heatmap_frames",
                            (0, stack.n_frames//2, stack.n_frames-1))
            z_um  = cfg.get("heatmap_z_um", float(np.array(stack).shape[0]))
            x_um  = cfg.get("heatmap_x_um", None)
            frs, phis, zc_list, xc_list = [], [], [], []
            for fi in want:
                stack.seek(int(fi))
                arr = np.array(stack, float)
                frs.append(arr)
                phis.append(fi * 0.1)
                tot = arr.sum()
                if tot > 0:
                    rr     = np.arange(arr.shape[0])
                    cc     = np.arange(arr.shape[1])
                    zc_px  = (rr[:, None] * arr).sum() / tot
                    xc_px  = (cc[None, :] * arr).sum() / tot
                    zc_list.append(zc_px / arr.shape[0] * z_um)
                    xc_list.append(xc_px if x_um is None
                                   else xc_px / arr.shape[1] * x_um)
                else:
                    zc_list.append(np.nan); xc_list.append(np.nan)
            heatmaps = {"frames": frs, "phis": phis, "z_um": z_um, "x_um": x_um,
                        "z_centroid_um": zc_list, "x_centroid_um": xc_list,
                        "path": tif_path}
            print(f"  Loaded heatmap TIFF: {tif_path}  "
                  f"({stack.n_frames} frames, showing {list(want)})")
        else:
            print(f"  WARNING: heatmap TIFF not found ({tif_path})")

    return dict(
        heights=heights, x_pf=x_pf, frames=frames_s, a_pf=a_pf, sa_pf=sa_pf, c_pf=c_pf, sc_pf=sc_pf,
        a_ok=a_ok, c_ok=c_ok, dof_pf=dof_pf,
        a_med=a_med, a_rsd=a_rsd, a_sem=a_sem, a_n=a_n,
        c_med=c_med, c_rsd=c_rsd, c_sem=c_sem, c_n=c_n,
        a_raw=a_raw,   c_raw=c_raw,   sa_raw=sa_raw,   sc_raw=sc_raw,
        a_gaus=a_gaus, c_gaus=c_gaus, sa_gaus=sa_gaus, sc_gaus=sc_gaus,
        angle_degs=angle_degs, z_cent=z_cent, z_max=z_max,
        I_tot=I_tot, I_cube=I_cube, row_um=row_um, col_um=col_um,
        PH=PH, fs_ph=fs_ph, fs_old=fs_old,
        twist_hkl=twist_hkl, heatmaps=heatmaps,
    )


# =============================================================================
# FIGURE + CSV EXPORT
# =============================================================================
def lattice_mean_per_height(z, vals, errs, ok, err_mode=LAT_MEAN_ERR):
    """
    Mean of the reliable scan points at each height (one row across the
    spicule). Error: standard error of the mean ("sem" = SD/sqrt(n)) or the
    spread of the points ("std"). With a single reliable point, its own fit
    error is used. Returns z_u, mean, err, n.
    """
    z = np.asarray(z, float); vals = np.asarray(vals, float)
    errs = np.asarray(errs, float); ok = np.asarray(ok, bool)
    z_u = np.unique(z[np.isfinite(vals)])
    mean = np.full(len(z_u), np.nan); err = np.full(len(z_u), np.nan)
    n = np.zeros(len(z_u), int)
    for i, zi in enumerate(z_u):
        sel = ok & (z == zi) & np.isfinite(vals)
        n[i] = int(sel.sum())
        if n[i] == 0:
            continue
        mean[i] = vals[sel].mean()
        if n[i] == 1:
            err[i] = errs[sel][0]
        else:
            sd = vals[sel].std(ddof=1)
            err[i] = sd / np.sqrt(n[i]) if err_mode == "sem" else sd
    return z_u, mean, err, n


def _lattice_panel(ax, z, x, vals, errs, ok, med, rsd, sym, row_step=5.0):
    """
    Panels a / b: median line and band, plus either
      LAT_PLOT_MODE = "mean": one point per height, the mean of that
                              height's reliable scan points (default), or
      LAT_PLOT_MODE = "all" : every reliable scan point, the points of one
                              height spread sideways by their x position.
    """
    if LAT_PLOT_MODE == "mean":
        z_u, mean, err, n = lattice_mean_per_height(z, vals, errs, ok)
        m = np.isfinite(mean)
        lab = ("mean per height (± s.e.m.)" if LAT_MEAN_ERR == "sem"
               else "mean per height (± SD)")
        ax.errorbar(z_u[m], mean[m], yerr=err[m], fmt="o", ms=3.6, lw=0,
                    elinewidth=0.7, capsize=1.8, color=C_DATA, mfc=C_DATA,
                    mec="white", mew=0.4, ecolor="0.45", label=lab, zorder=3)
        heights = np.asarray(z, float)
        ok_plot = np.zeros(len(heights), bool)   # skip the scatter below
    else:
        x = np.asarray(x, float)
        xr = (x - x.min()) / (np.ptp(x) if np.ptp(x) > 0 else 1.0) - 0.5
        heights = np.asarray(z, float) + xr * LAT_X_SPREAD * row_step
        ok_plot = ok
    if SHOW_REJECTED and LAT_PLOT_MODE != "mean":
        rej = np.isfinite(vals) & ~ok
        ax.plot(heights[rej], vals[rej], "o", ms=2.6, mfc="none",
                mec=C_EXCL, mew=0.5, zorder=2, label="rejected")
    if ok_plot.any():
        ax.errorbar(heights[ok_plot], vals[ok_plot], yerr=errs[ok_plot],
                    fmt="o", ms=2.6, lw=0, elinewidth=0.5, capsize=1.2,
                    color=C_DATA, mfc=C_DATA, mec="white", mew=0.3,
                    ecolor="0.55", label="per-position fit", zorder=3)
    if np.isfinite(med):
        ax.axhline(med, color=C_GAUS, lw=1.3, zorder=4,
                   label=f"median  ${sym}$ = {med:.2f} \u00b1 {rsd:.2f} \u00c5")
        ax.axhspan(med - rsd, med + rsd, color=C_GAUS, alpha=0.15, zorder=1)
    ax.set_xlabel("Position along the spicule ($\\mu$m)")
    ax.set_ylabel(f"${sym}$ (\u00c5)")
    zt = np.unique(np.asarray(z, float)[np.isfinite(vals)])
    ax.set_xlim(zt.min() - 0.6*row_step, zt.max() + 0.6*row_step)
    ax.set_xticks(zt[::2] if len(zt) > 6 else zt)
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(zt))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 1.01), ncol=3,
              columnspacing=1.4, borderaxespad=0, frameon=False)


def build_figure(R, cfg, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    set_nature_style()
    species = cfg["species"]
    hkl_str = "".join(str(i) for i in R["twist_hkl"])

    fig = plt.figure(figsize=(FIG_W_MM*MM, FIG_H_MM*MM))
    gs  = fig.add_gridspec(
        4, 6,
        height_ratios=[1.0, 1.0, 1.05, 1.05],
        hspace=0.62, wspace=1.0,
        left=0.10, right=0.95, top=0.955, bottom=0.075)

    ax_a    = fig.add_subplot(gs[0, 0:6])
    ax_c    = fig.add_subplot(gs[1, 0:6])
    ax_g0   = fig.add_subplot(gs[2, 0:2])
    ax_g1   = fig.add_subplot(gs[2, 2:4])
    ax_g2   = fig.add_subplot(gs[2, 4:6])
    ax_itot = fig.add_subplot(gs[3, 0:3])
    if SHOW_RESIDUALS:
        sg      = gs[3, 3:6].subgridspec(2, 1, height_ratios=[2.6, 1],
                                         hspace=0.08)
        ax_zfit = fig.add_subplot(sg[0])
        ax_res  = fig.add_subplot(sg[1], sharex=ax_zfit)
    else:
        ax_zfit = fig.add_subplot(gs[3, 3:6])
        ax_res  = None

    def _panel_letter(ax_obj, letter):
        ax_obj.text(-0.02, 1.10, letter, transform=ax_obj.transAxes,
                    fontsize=10, fontweight="bold", va="bottom", ha="right")

    # ---------- Panels a, b: lattice parameters ----------
    rstep = float(np.abs(np.diff(R["row_um"])).mean()) if len(R["row_um"]) > 1 else 5.0
    _lattice_panel(ax_a, R["heights"], R["x_pf"], R["a_pf"], R["sa_pf"],
                   R["a_ok"], R["a_med"], R["a_rsd"], "a", rstep)
    ax_a.set_title(species, loc="right", fontstyle="italic", fontsize=7,
                   color="0.4")
    _panel_letter(ax_a, "a")
    _lattice_panel(ax_c, R["heights"], R["x_pf"], R["c_pf"], R["sc_pf"],
                   R["c_ok"], R["c_med"], R["c_rsd"], "c", rstep)
    _panel_letter(ax_c, "b")

    # ---------- Panels c/d/e: heatmaps ----------
    hm      = R.get("heatmaps")
    hm_axes = [ax_g0, ax_g1, ax_g2]
    if hm is not None:
        frames = hm["frames"]; phis = hm["phis"]
        z_um   = hm["z_um"];   x_um = hm["x_um"]
        x_ext  = x_um if x_um is not None else float(frames[0].shape[1])
        for ax_g, fr, phi, let in zip(hm_axes, frames, phis, ["c", "d", "e"]):
            disp = fr.astype(float)
            pos  = disp[disp > 0]
            vmax = np.percentile(pos, 99.5) if pos.size else 1.0
            im_g = ax_g.imshow(disp, aspect="auto", origin="upper",
                               cmap="inferno", vmin=0, vmax=max(vmax, 1e-9),
                               extent=[0, x_ext, z_um, 0],
                               interpolation="lanczos")
            im_g.set_rasterized(True)
            ax_g.set_title(f"$\\varphi$ = {phi:.1f}\u00b0", fontsize=7,
                           color="black", loc="center", pad=3)
            ax_g.set_xlabel("Position across spicule ($\\mu$m)")
            ax_g.set_ylabel("Position along spicule ($\\mu$m)")
            ax_g.set_xlim(0, x_ext)
            ax_g.set_ylim(z_um, 0)
            ax_g.set_xticks([0, x_ext/2, x_ext])
            ax_g.set_xticklabels([f"0", f"{x_ext/2:.0f}", f"{x_ext:.0f}"])
            _panel_letter(ax_g, let)

    # ---------- Panel f: intensity histogram vs phi for each height ----------
    ax = ax_itot
    PH = R["PH"]; fs = R["fs_ph"]
    phi_it = R["angle_degs"]
    dphi = np.diff(phi_it).min() if len(phi_it) > 1 else 0.1
    z_all = PH["z_um"]
    # red-free spectrum, spread over the heights used in the twist fit
    cmap  = mpl.colors.LinearSegmentedColormap.from_list("heights", HEIGHT_RAMP)
    z_use = z_all[PH["used"]] if PH["used"].any() else z_all
    znorm = mpl.colors.Normalize(vmin=z_use.min(), vmax=z_use.max())
    phi_fine = np.linspace(phi_it.min(), phi_it.max(), 600)
    shown_extra = show_in_f(PH, dphi, phi_it.min(), phi_it.max())
    for r in range(len(z_all)):
        y = PH["I_row"][:, r]
        if y.max() <= 0:
            continue
        used_r  = PH["used"][r]
        extra_r = shown_extra[r]
        if not (used_r or extra_r):
            continue
        # heights in the twist fit: coloured by height;
        # weak / edge-cut heights with a real Gaussian: grey, open marker
        col = cmap(znorm(z_all[r])) if used_r else C_EXCL
        mu_r, sd_r = PH["phi_peak_deg"][r], PH["phi_std_deg"][r]
        # measured intensity at each scanned angle around this peak
        if np.isfinite(mu_r) and np.isfinite(sd_r):
            win = np.abs(phi_it - mu_r) <= max(4*sd_r, 3*dphi)
            ax.plot(phi_it[win], y[win], "o", ms=2.6, color=col,
                    alpha=0.85 if used_r else 0.5, mew=0, zorder=2)
        if PH["method"][r] == "gauss":
            pf = phi_fine[np.abs(phi_fine - mu_r) <= 4*sd_r]
            g = gaussian_bg(pf, PH["A"][r], mu_r, sd_r, PH["bg"][r])
            ax.plot(pf, g, "--" if used_r else ":", color=col,
                    lw=1.0 if used_r else 0.9, zorder=3 if used_r else 2)
    ax.set_xlim(phi_it.min()-dphi*0.6, phi_it.max()+dphi*0.6)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.08)
    ax.set_xlabel("Rotation angle $\\varphi$ (\u00b0)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # key above the panel (neutral grey proxies; colours follow height)
    from matplotlib.lines import Line2D
    # key above the panel: symbols + one entry per height (colour = height)
    key = [Line2D([], [], ls="none", marker="o", ms=2.6, color="0.45",
                  alpha=0.85, mew=0, label="measured"),
           Line2D([], [], ls="--", lw=1.0, color="0.45", label="Gaussian fit")]
    for r in np.argsort(z_all):
        if PH["used"][r] and np.isfinite(PH["phi_peak_deg"][r]):
            key.append(Line2D([], [], ls="-", lw=1.6,
                              color=cmap(znorm(z_all[r])),
                              label=f"{z_all[r]:g} $\\mu$m"))
    ax.legend(handles=key, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=6, borderaxespad=0, handlelength=1.3, handletextpad=0.4,
              columnspacing=0.9, fontsize=5.5, frameon=False,
              alignment="left")
    _panel_letter(ax, "f")

    # ---------- Panel g: peak angle +/- std per height, weighted fit ----------
    ax  = ax_zfit
    z   = PH["z_um"]; ph = PH["phi_peak_deg"]; er = PH["phi_err_deg"]
    fin = np.isfinite(ph)
    use = fin & PH["used"]
    # excluded heights are only drawn if they have a real peak (e.g. at the
    # scan edge); pure-noise heights have no meaningful peak angle
    exc = fin & show_in_f(PH, dphi, phi_it.min(), phi_it.max())
    err_lab = "std" if PH["err_mode"] == "std" else "s.e."
    # In the unweighted fit every height has the same uncertainty, estimated
    # from the scatter about the line (residual rms); show that as the bar.
    if G_ERRBAR == "gauss":
        # each point: its own Gaussian-fit error of the centre (panel f)
        er = PH["phi_mu_err_deg"]
    elif fs is not None and not fs["weighted"]:
        er = np.where(use, fs["resid_rms_deg"], np.nan)
        err_lab = "1$\\sigma$ from fit scatter"
    # axes: x = position along the spicule, y = rotation angle (both increase
    # from the origin outward); the fit is phi = slope*z + intercept
    if fs is not None:
        zf  = np.linspace(fs["z_min"], fs["z_max"], 100)
        phf = fs["slope"]*zf + fs["intercept"]
        bd  = fit_band(fs, zf)
        ax.fill_between(zf, phf - bd, phf + bd, color=C_FIT, alpha=0.18,
                        lw=0, zorder=2)
        ax.plot(zf, phf, "-", color=C_FIT, lw=1.4, zorder=3,
                label=(f"{'weighted ' if fs['weighted'] else ''}linear fit  "
                       f"twist = {fs['slope']:.4f} "
                       f"\u00b1 {fs['stderr']:.{4 if fs['stderr'] >= 1e-3 else 5}f} \u00b0/$\\mu$m  "
                       f"$R^2$ = {fs['r2']:.4f}"))
    # each height in the same colour as its Gaussian in panel f
    for r in np.where(use)[0]:
        ax.errorbar(z[r], ph[r], yerr=er[r], fmt="o", ms=4,
                    color=cmap(znorm(z[r])), mec="white", mew=0.4,
                    elinewidth=0.7, capsize=1.8, ecolor="0.45", zorder=4)
    # neutral legend entry (the point colours just follow panel f)
    ax.plot([], [], "o", ms=4, color="0.45", mec="white", mew=0.4,
            label="peak angle of each height")
    if exc.any():
        ax.scatter(z[exc], ph[exc], s=16, facecolors="none",
                   edgecolors=C_EXCL, linewidths=0.7, zorder=4,
                   label="not in fit (peak at/outside scan edge)")
    zr = R["row_um"]
    ax.set_xlim(zr.min() - 2, zr.max() + 2)
    ax.set_ylim(phi_it.min()-dphi*0.6, phi_it.max()+dphi*0.6)
    ax.set_xlabel("Position along spicule ($\\mu$m)")
    ax.set_ylabel("Rotation angle $\\varphi$ (\u00b0)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.005, 1.01),
              ncol=1, borderaxespad=0, handlelength=1.6,
              columnspacing=1.2, frameon=False)
    _panel_letter(ax, "g")

    if ax_res is not None:
        # ---- residual strip: measured - fitted peak angle, vs position ----
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        ar = ax_res
        ar.axhline(0, color=C_FIT, lw=0.9, zorder=3)
        if fs is not None:
            ar.fill_between(zf, -bd, bd, color=C_FIT, alpha=0.18, lw=0,
                            zorder=2)
            res = ph - (fs["slope"]*z + fs["intercept"])
            ar.errorbar(z[use], res[use], yerr=er[use], fmt="o", ms=3.2,
                        color=C_CENT, mec="white", mew=0.4, elinewidth=0.7,
                        capsize=1.6, ecolor="0.45", zorder=4)
            lim = np.nanmax(np.abs(np.concatenate(
                [res[use] + er[use], res[use] - er[use], bd])))
            ar.set_ylim(-1.25*lim, 1.25*lim)
        ar.set_xlabel("Position along spicule ($\\mu$m)")
        ar.set_ylabel("Resid. (\u00b0)")
        ar.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, symmetric=True))
        ar.xaxis.set_minor_locator(AutoMinorLocator(2))
        fig.align_ylabels([ax, ar])

    # ---------- Save figure ----------
    save_dpi = 600
    for ext in ("pdf", "svg"):
        out = os.path.join(outdir, f"{tag}_supplementary.{ext}")
        fig.savefig(out, dpi=save_dpi, facecolor="white")
        print(f"  FIG : {out}  (vector, rasters @ {save_dpi} dpi)")
    for png_dpi in (600, 1200):
        suffix = "" if png_dpi == 600 else f"_{png_dpi}dpi"
        out = os.path.join(outdir, f"{tag}_supplementary{suffix}.png")
        fig.savefig(out, dpi=png_dpi, facecolor="white")
        print(f"  FIG : {out}  ({png_dpi} dpi)")
    try:
        out = os.path.join(outdir, f"{tag}_supplementary.tif")
        fig.savefig(out, dpi=save_dpi, facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        print(f"  FIG : {out}  (600 dpi TIFF)")
    except Exception as e:
        print(f"  (TIFF export skipped: {e})")
    plt.close(fig)

    # ---------- CSV exports ----------
    export_csv(
        os.path.join(outdir, f"{tag}_panelA_lattice_vs_position.csv"),
        ["frame", "z_along_spicule_um", "x_across_spicule_um",
         "a_Ang", "a_err_Ang", "a_reliable",
         "c_Ang", "c_err_Ang", "c_reliable", "dof"],
        [[int(v) for v in R["frames"]], list(R["heights"]), list(R["x_pf"]),
         list(R["a_pf"]), list(R["sa_pf"]),
         [int(v) for v in R["a_ok"]],
         list(R["c_pf"]), list(R["sc_pf"]),
         [int(v) for v in R["c_ok"]], [int(v) for v in R["dof_pf"]]])

    za, ma, ea, na = lattice_mean_per_height(R["heights"], R["a_pf"],
                                             R["sa_pf"], R["a_ok"])
    zc, mc, ec, nc = lattice_mean_per_height(R["heights"], R["c_pf"],
                                             R["sc_pf"], R["c_ok"])
    export_csv(
        os.path.join(outdir, f"{tag}_panelAB_lattice_mean_per_height.csv"),
        ["z_along_spicule_um", "a_mean_Ang", f"a_{LAT_MEAN_ERR}_Ang", "a_n",
         "c_mean_Ang", f"c_{LAT_MEAN_ERR}_Ang", "c_n"],
        [list(za), list(ma), list(ea), [int(v) for v in na],
         list(mc), list(ec), [int(v) for v in nc]])

    export_csv(
        os.path.join(outdir, f"{tag}_panelA_global_values.csv"),
        ["method", "a_Ang", "a_err_Ang", "c_Ang", "c_err_Ang", "n_points"],
        [["median_reliable (err = robust SD)",
          "median_reliable (err = SE of median)",
          "raw_maxI", "gaussian_fit"],
         [R["a_med"], R["a_med"], R["a_raw"], R["a_gaus"]],
         [R["a_rsd"], R["a_sem"], R["sa_raw"], R["sa_gaus"]],
         [R["c_med"], R["c_med"], R["c_raw"], R["c_gaus"]],
         [R["c_rsd"], R["c_sem"], R["sc_raw"], R["sc_gaus"]],
         [f"a:{R['a_n']} c:{R['c_n']}", f"a:{R['a_n']} c:{R['c_n']}", "", ""]])

    # Panel f data (+ old centroid, for reference only)
    export_csv(
        os.path.join(outdir, f"{tag}_panelF_intensity_vs_angle.csv"),
        ["phi_deg", "I_total", "z_centroid_um_OLD", "z_maxpix_um_OLD"],
        [list(R["angle_degs"]), list(R["I_tot"]),
         list(R["z_cent"]), list(R["z_max"])])

    # Panel g data: one row per height
    PH = R["PH"]
    export_csv(
        os.path.join(outdir, f"{tag}_panelG_peak_angle_per_height.csv"),
        ["z_um", "phi_peak_deg", "phi_std_deg", "phi_pos_err_deg",
         "phi_err_used_deg", "gauss_amplitude", "gauss_background",
         "peak_method", "I_peak", "snr", "used_in_fit", "note"],
        [list(PH["z_um"]), list(PH["phi_peak_deg"]), list(PH["phi_std_deg"]),
         list(PH["phi_mu_err_deg"]), list(PH["phi_err_deg"]),
         list(PH["A"]), list(PH["bg"]), PH["method"], list(PH["I_peak"]),
         [round(float(v), 2) for v in PH["snr"]],
         [int(v) for v in PH["used"]], PH["note"]])

    # Panel f data: per-height intensity histograms vs angle
    export_csv(
        os.path.join(outdir, f"{tag}_panelF_histogram_per_height_vs_angle.csv"),
        ["phi_deg"] + [f"I_z{z:.1f}um" for z in PH["z_um"]],
        [list(R["angle_degs"])] + [list(PH["I_row"][:, r])
                                   for r in range(PH["I_row"].shape[1])])

    rows = []
    for ai, a in enumerate(R["angle_degs"]):
        for r in range(R["I_cube"].shape[1]):
            for c in range(R["I_cube"].shape[2]):
                rows.append((a, R["row_um"][r], R["col_um"][c],
                             R["I_cube"][ai, r, c]))
    with open(os.path.join(outdir, f"{tag}_intensity_cube.csv"),
              "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["phi_deg", "z_um", "x_um", "I_max"])
        wr.writerows(rows)
    print(f"  CSV : {tag}_intensity_cube.csv")

    hm = R.get("heatmaps")
    if hm is not None:
        for fr, phi in zip(hm["frames"], hm["phis"]):
            H, W    = fr.shape
            z_axis  = np.linspace(0, hm["z_um"], H)
            x_um_hm = hm["x_um"]
            if x_um_hm is not None:
                x_axis = np.linspace(0, x_um_hm, W); xlab = "x_um"
            else:
                x_axis = np.arange(W); xlab = "x_px"
            fname = os.path.join(
                outdir, f"{tag}_panelCDE_heatmap_phi{phi:.1f}deg.csv")
            with open(fname, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["z_um", xlab, "I_100"])
                for iz in range(H):
                    for ix in range(W):
                        if fr[iz, ix] > 0:
                            wr.writerow([f"{z_axis[iz]:.4f}",
                                         f"{x_axis[ix]:.4f}",
                                         f"{fr[iz, ix]:.4f}"])
            print(f"  CSV : {os.path.basename(fname)}")
        export_csv(
            os.path.join(outdir, f"{tag}_panelCDE_heatmap_centroids.csv"),
            ["phi_deg", "z_centroid_um", "x_centroid"],
            [list(hm["phis"]), list(hm["z_centroid_um"]),
             list(hm["x_centroid_um"])])

    # ---------- Caption ----------
    fs = R.get("fs_ph")
    tw = tw_err = tw_prop = chi2r = np.nan; n_fit = 0
    if fs is not None:
        tw, tw_err, n_fit = fs["slope"], fs["stderr"], fs["n"]
        tw_prop, chi2r    = fs["stderr_propagated"], fs["chi2_red"]
    PHc = R["PH"]
    prof_txt = ("summed over the spicule width" if PHc["profile"] == "sum"
                else "taken as the brightest pixel across the spicule")
    err_txt  = ("the standard deviation (width) of the Gaussian peak"
                if PHc["err_mode"] == "std"
                else ("the 1 sigma uncertainty of each peak angle: the Gaussian-fit\n"
                      f"error of the centre and the rotation-motor uncertainty\n"
                      f"({MOTOR_ERR_DEG:g} deg) added in quadrature")
                if MOTOR_ERR_DEG > 0
                else "the Gaussian-fit error of the peak centre")
    if G_ERRBAR == "gauss":
        err_txt = ("the 1 sigma uncertainty of each peak centre from its Gaussian\n"
                   "fit in (f)")
    elif fs is not None and not fs["weighted"]:
        err_txt = (f"the 1 sigma uncertainty of a single peak angle estimated "
                   f"from the scatter\nabout the fit ({fs['resid_rms_deg']:.3f} deg)")
    if fs is not None and PHc["err_mode"] == "gauss":
        fit_txt = (
            f"The red line is a weighted least-squares fit (weights 1/error^2) of\n"
            f"peak angle vs height over {n_fit} heights. The twist uncertainty is\n"
            f"propagated from the Gaussian-fit errors of the peak centres only\n"
            f"(reduced chi^2 = {chi2r:.2f}). Estimated instead from the scatter of the\n"
            f"points about the line, which also includes e.g. rotation-stage\n"
            f"positioning, it is {fs['stderr_scatter']:.4f} deg/um.")
    elif fs is not None and fs["weighted"]:
        fit_txt = (
            f"The red line is a weighted least-squares fit (weights 1/error^2) of\n"
            f"peak angle vs height over {n_fit} heights; the per-height errors are\n"
            f"propagated through the fit covariance (reduced chi^2 = {chi2r:.2f}"
            + ("; the error is scaled by sqrt(chi^2_red)" if chi2r > 1 else
               ", i.e. the errors fully account for the scatter")
            + ").")
    else:
        fit_txt = (
            f"The red line is an unweighted linear least-squares fit of peak angle\n"
            f"vs height over {n_fit} heights. "
            + (f"Because the Gaussian-fit uncertainties of the peak\n"
               f"centres do not account for the observed scatter (reduced chi^2 =\n"
               f"{chi2r:.2f}; e.g. rotation-stage positioning is not included), the\n"
               if np.isfinite(chi2r) else "The ")
            + f"uncertainty of the slope is the standard error estimated from the fit\n"
            f"residuals (residual rms {fs['resid_rms_deg'] if fs else np.nan:.3f} deg); "
            f"R^2 = {fs['r2'] if fs else np.nan:.4f}.")
    n_rows_tot = len(R["row_um"])
    n_a    = int(np.isfinite(R["a_pf"]).sum())
    n_c    = int(np.isfinite(R["c_pf"]).sum())
    hm_x   = int(R["heatmaps"]["x_um"]) if R.get("heatmaps") and R["heatmaps"]["x_um"] else 4
    hm_z   = int(R["heatmaps"]["z_um"]) if R.get("heatmaps") else int(cfg.get("heatmap_z_um", 50))
    hm_phis = R["heatmaps"]["phis"] if R.get("heatmaps") else [0, 3, 6]
    row_step = float(np.abs(np.diff(R["row_um"])).mean()) if len(R["row_um"]) > 1 else 0
    dphi_c   = float(np.median(np.diff(R["angle_degs"]))) if len(R["angle_degs"]) > 1 else 0.1
    shown_ex = show_in_f(PHc, dphi_c, R["angle_degs"].min(), R["angle_degs"].max())
    hidden   = sorted(z for z, u, sh in zip(PHc["z_um"], PHc["used"], shown_ex)
                      if not u and not sh)
    hidden_s = [f"{z:g}" for z in hidden]
    hidden_s = (" and ".join(hidden_s) if len(hidden_s) <= 2
                else ", ".join(hidden_s[:-1]) + " and " + hidden_s[-1])
    hidden_txt = (f"{'Heights' if len(hidden) > 1 else 'Height'} {hidden_s} um "
                  f"{'are' if len(hidden) > 1 else 'is'} not shown and not used in the fit: "
                  f"{'they have' if len(hidden) > 1 else 'it has'}\n"
                  f"only weak signal with no peak inside the scanned angular range (at most\n"
                  f"the tail of a peak outside it).\n"
                  if hidden else "")
    if shown_ex.any():
        grey_txt = ("Grey dotted curves: heights not used in the twist\n"
                    "fit because their signal is weak and the peak centre lies at or beyond\n"
                    "the edge of the angular scan (only the tail is measured); shown for\n"
                    "completeness. " + hidden_txt + "\n")
    else:
        grey_txt = hidden_txt + "\n"
    cap = (
        f"Supplementary Figure | Lattice parameters and lattice twist of the\n"
        f"{species} spicule (100) reflection, from serial synchrotron micro-diffraction.\n\n"
        f"(a) Hexagonal lattice parameter a (= b) refined at each scan position\n"
        f"by intensity-weighted least-squares of 1/d^2 against the indexed\n"
        f"reflections, plotted against the position along the spicule\n"
        f"({len(R['row_um'])} heights, {row_step:g} um apart). "
        + (f"Each blue point is the mean of\nthe reliable scan positions across the "
           f"spicule at that height, with error bars\ngiving the "
           + ("standard error of the mean" if LAT_MEAN_ERR == "sem"
              else "standard deviation of those positions")
           + f".\n{R['a_n']} of {n_a} scan positions were reliable;"
           if LAT_PLOT_MODE == "mean" else
           f"Points from the same height\nare spread horizontally in order of their "
           f"position across the spicule for visibility only.\n"
           f"Blue points are the reliable per-position fits ({R['a_n']} of {n_a})\n"
           f"with 1 sigma error bars;")
        + f" positions with fewer than {LAT_MIN_DOF} degrees of\n"
        f"freedom, sigma > {100*(LAT_MAX_REL_SIG or 0):g}% of the value or > {LAT_K_SIGMA:g}x "
        f"the median sigma, or values\nmore than {LAT_K_MAD:g} robust SD "
        f"from the median were rejected.\n"
        f"The orange line and band give the median a = {R['a_med']:.2f} +/- "
        f"{R['a_rsd']:.2f} A (robust SD, 1.4826 x MAD).\n\n"
        f"(b) As (a) for the c parameter ({R['c_n']} of {n_c} positions): "
        f"median c = {R['c_med']:.2f} +/- {R['c_rsd']:.2f} A.\n\n"
        f"(c-e) Real-space maps of the (100) reflected intensity over the scanned grid\n"
        f"(horizontal = position across the spicule, 0-{hm_x} um;\n"
        f"vertical = position along the spicule, 0-{hm_z} um) at sample rotation angles\n"
        f"phi = {hm_phis[0]:.1f} deg (c), {hm_phis[1]:.1f} deg (d) and {hm_phis[2]:.1f} deg (e).\n\n"
        f"(f) (100) intensity vs rotation angle for each of the {n_rows_tot} scanned heights\n"
        f"({row_step:g} um apart; intensity {prof_txt}). Dashed curves: Gaussian\n"
        f"fits to each height's intensity histogram, coloured by height; dots: the\n"
        f"measured intensity at each scanned angle; the key above gives the position\n"
        f"along the spicule for each colour.\n"
        f"{grey_txt}"
        f"(g) Fitted peak angle vs position along the spicule for each height used in\n"
        f"the fit, coloured as in (f); vertical bars are\n"
        f"{err_txt}.\n"
        + ("Open grey symbols: heights shown in (f) but not used in the fit; their angle\n"
           "marks only where the scan ends, not the true peak position.\n"
           if shown_ex.any() else "")
        + f"{fit_txt}\n"
        f"The shaded band is the 1 sigma confidence interval of the fit. "
        + ("The lower strip\nshows the residuals (measured minus fitted peak angle) "
           "with the same error\nbars and confidence band. "
           if SHOW_RESIDUALS else "")
        + f"The twist is\n"
        f"the slope: {tw:.4f} +/- {tw_err:.{4 if tw_err >= 1e-3 else 5}f} deg/um (1 sigma); the sign gives\n"
        f"the sense of rotation (positive: the peak angle increases with position along\n"
        f"the spicule).\n"
    )
    cap_path = os.path.join(outdir, f"{tag}_caption.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(cap)
    print(f"  TXT : {cap_path}")

    # ---------- Fit summary JSON ----------
    twist_json = {}
    if fs is not None:
        twist_json["peak_angle_per_height"] = {
            "slope_deg_per_um":     float(fs["slope"]),
            "twist_deg_per_um":     float(fs["slope"]),   # signed: d(phi)/dz
            "twist_err_deg_per_um": float(fs["stderr"]),
            "twist_err_propagated_deg_per_um": float(fs["stderr_propagated"]),
            "twist_err_scatter_deg_per_um": float(fs["stderr_scatter"]),
            "chi2":                 float(fs["chi2"]),
            "dof":                  int(fs["dof"]),
            "chi2_red":             float(fs["chi2_red"]),
            "chi2_probability":     float(fs["p_chi2"]),
            "fit_weighting":        fs["weighting"],
            "residual_rms_deg":     float(fs["resid_rms_deg"]),
            "intercept_deg":        float(fs["intercept"]),
            "intercept_err_deg":    float(fs["intercept_err"]),
            "pitch_um":             float(360.0/abs(fs["slope"]))
                                    if fs["slope"] != 0 else None,
            "r2":                   float(fs["r2"]),
            "peak_profile":         R["PH"]["profile"],
            "peak_error_mode":      R["PH"]["err_mode"],
            "n_heights":            fs["n"],
            "z_range_um":           [fs["z_min"], fs["z_max"]],
        }
    fo = R.get("fs_old")
    if fo is not None and fo["slope"] != 0:
        twist_json["old_centroid_all_points_REFERENCE_ONLY"] = {
            "twist_deg_per_um": float(1.0/abs(fo["slope"])),
            "r2":               float(fo["r2"]),
            "n_points":         fo["n"],
        }
    fit_summary = {
        "sample":  tag, "species": species,
        "lattice_global_median": {
            "a": R["a_med"], "a_robust_sd": R["a_rsd"], "a_se_median": R["a_sem"],
            "a_n_reliable": R["a_n"],
            "c": R["c_med"], "c_robust_sd": R["c_rsd"], "c_se_median": R["c_sem"],
            "c_n_reliable": R["c_n"]},
        "lattice_global_gauss": {
            "a": R["a_gaus"], "a_err": R["sa_gaus"],
            "c": R["c_gaus"], "c_err": R["sc_gaus"]},
        "twist_hkl":    hkl_str,
        "twist_method": "per-height intensity histogram vs angle -> Gaussian "
                        "peak +/- std -> weighted linear fit phi vs z with "
                        "error propagation",
        "twist":        twist_json,
    }
    with open(os.path.join(outdir, f"{tag}_fit_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(fit_summary, f, indent=2)
    print(f"  JSON: {tag}_fit_summary.json")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    tags = list(SAMPLES) if ACTIVE_TAG == "ALL" else [ACTIVE_TAG]
    summary = []
    for tag in tags:
        cfg = SAMPLES[tag]
        print(f"\nProcessing {tag}  ({cfg['species']}) ...")
        try:
            R = run_analysis(cfg)
            build_figure(R, cfg, tag, OUTPUT_DIR)
            fs = R.get("fs_ph")
            summary.append((tag, cfg["species"],
                            f"{fs['slope']:+.4f} +/- {fs['stderr']:.4f} deg/um "
                            f"(n = {fs['n']})" if fs else "no twist fit"))
        except Exception as e:          # keep going with the other species
            print(f"  ERROR in {tag}: {e}")
            summary.append((tag, cfg["species"], f"FAILED: {e}"))
    if len(tags) > 1:
        print("\nSummary")
        for tag, sp, res in summary:
            print(f"  {tag:10s} {sp:24s} {res}")
    print("Done.")
