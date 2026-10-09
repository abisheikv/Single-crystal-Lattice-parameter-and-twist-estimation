# Lattice parameters and single-crystal twist from rotation nano-diffraction

Analysis and figure script for synchrotron X-ray nano-diffraction raster scans
recorded at a series of sample rotation angles. It was developed for the
hexagonal (silicatein) protein crystals in demosponge spicules, but applies to
any elongated single crystal scanned along its length. For each dataset it
measures

* the **hexagonal lattice parameters** *a* and *c* along the crystal, and
* the **lattice twist rate**: the rotation of the crystal lattice per µm along
  the crystal, from a chosen reflection (default (100)),

and produces a publication-style supplementary figure, CSV files of every
plotted series, a fit summary (JSON) and a draft figure caption.

Datasets configured in the script (the tags match the folder and file names
of the original measurements; add your own in the `SAMPLES` block):

| Tag | Species |
|---|---|
| `TA` | *Tethya aurantium* |
| `SP` | *Stryphnus ponderosus* |
| `RP` | *Rhizaxinella pyrifera* |
| `GC` | *Geodia cydonium* |
| `PW` | *Clathria prolifera* |

---

## The figure

| Panel | Content |
|---|---|
| **a, b** | Lattice parameters *a* and *c* vs position along the crystal. Points: mean of the reliable scan positions at each height (± SD). Line and band: median ± robust SD of all reliable positions. |
| **c–e** | Real-space maps of the (100) intensity over the scanned grid at three rotation angles. |
| **f** | (100) intensity vs rotation angle for each scanned height (dots), with a Gaussian fit per height (dashed). Colour = position along the spicule. |
| **g** | Fitted peak angle of each height vs position along the spicule, with a straight-line fit. The slope is the twist rate; the legend gives twist ± 1σ and R². |

---

## Method

### Measurement geometry
Each dataset is a raster scan (default 11 rows × 9 columns: 5 µm steps along
the spicule, 0.5 µm across it) repeated at a series of sample rotation angles
φ (default 0–6° in 0.1° steps).

### Lattice parameters (panels a, b)
1. At every scan position, all indexed reflections from all rotation angles
   are pooled; for each *hkl* the strongest observation is kept.
2. *a* and *c* are refined by intensity-weighted least squares of
   1/d² = 4/3·(h² + hk + k²)/a² + l²/c².
3. Positions are screened for reliability (finite error, enough degrees of
   freedom, error not much larger than typical, value within *k*·MAD of the
   median; see `LAT_*` settings and the per-sample `lat_screen` overrides).
   The console prints why positions were rejected.
4. Reported value: median ± robust SD (1.4826·MAD) of the reliable positions.

### Twist (panels f, g)
1. For each scanned height, the (100) intensity is summed across the spicule
   at every rotation angle, giving one intensity-vs-angle profile (rocking
   curve) per height.
2. Each profile is fitted with a Gaussian plus constant background in a
   ±0.8° window around its maximum. The Gaussian centre μ is the **peak
   angle** of that height.
3. A height is used only if it has a real peak: signal-to-noise ≥ 5, peak
   intensity ≥ 5 % of the brightest height, a resolved width, and a centre
   that is not pinned to the first/last scanned angle (i.e. the peak lies
   inside the angular scan, not just its tail).
4. The twist is the slope of a straight-line fit
   **φ_peak = twist · z + const**, fitted directly (no inversion). The sign of
   the twist gives the sense of rotation.
5. Uncertainty (default `PEAK_ERR_MODE = "residual"`): ordinary least-squares
   standard error of the slope, estimated from the scatter of the peak angles
   about the line. The Gaussian-fit errors of the individual peak centres are
   shown as error bars in panel g and saved in the CSV; they are typically far
   smaller than the observed scatter (reduced χ² ≫ 1) because they do not
   include e.g. rotation-stage positioning, so they are not used for the twist
   uncertainty by default. Other options: `"gauss"` (propagate the
   Gaussian-fit errors only), `"auto"` (χ² test decides), `"sem"`, `"std"`.

#### Why not the intensity-weighted centroid?
An earlier version tracked the intensity-weighted centroid of the reflection
along the spicule at each angle. With heights 5 µm apart, the centroid can only
sit on (or between) scanned rows, which produces a staircase in the
position-vs-angle plot. This is a sampling artefact, not crystal structure;
the per-height peak-angle method above avoids it.

---

## Input data

### Peak tables
One folder per dataset under `DATA_ROOT`:

```
DATA_ROOT/
  TAEshelby/                             # data folder of dataset "TA"
    0/                                   # rotation angle (tenths of a degree)
      TAeshelby_0_<n>_data_000001_000000/
        TAeshelby_0_<n>_data_000001_000000_peaks_lattice.csv
      ...                                # one folder per scan frame (0..98)
    1/
    ...
```

Each `*_peaks_lattice.csv` has a header line and at least 10 comma-separated
columns; the script reads

| Column (0-based) | Content |
|---|---|
| 0, 1, 2 | h, k, l |
| 5 | peak intensity |
| 6 | measured d-spacing (Å) |
| 9 | relative deviation (read but not used) |

Folder and file naming is set per dataset (`prefix_re`, `prefix_fmt`,
`angles`, `angle_offset`) in the `SAMPLES` block.

### Heat-map TIFF stacks (panels c–e, optional)
Multi-frame TIFF of the (100) intensity map, one frame per rotation angle,
path given by `heatmap_tif` relative to `HEATMAP_ROOT`. The frames shown are
set by `heatmap_frames`. If the file is missing, panels c–e stay empty and the
rest of the analysis still runs.

---

## Installation

Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

## Usage

1. Point the script to your data, either by editing the two lines near the top
   of `lattice_twist_analysis.py`

   ```python
   DATA_ROOT    = r"T:\path\to\raw_data"
   HEATMAP_ROOT = r"W:\path\to\heatmaps"
   ```

   or by setting environment variables:

   ```bash
   export LATTICE_DATA_ROOT=/path/to/raw_data
   export LATTICE_HEATMAP_ROOT=/path/to/heatmaps
   ```

2. Choose the dataset: `ACTIVE_TAG = "GC"` (one) or `ACTIVE_TAG = "ALL"`
   (all five; a twist summary is printed at the end, and a failing dataset
   does not stop the others).

3. Run:

   ```bash
   python lattice_twist_analysis.py
   ```

To add a dataset, copy one block in `SAMPLES` and adapt the folder name,
naming pattern, angle list, scan geometry and the reflection used for the
twist (`twist_hkl`). Other lattice systems need a different
`weighted_lstsq_hexagonal` (the 1/d² relation is hexagonal).

## Output

Written to `supplementary_output/` (per dataset tag):

| File | Content |
|---|---|
| `<tag>_supplementary.{pdf,svg,png,tif}` (+ `_1200dpi.png`) | The figure (vector + 600/1200 dpi) |
| `<tag>_caption.txt` | Draft figure caption with the fitted numbers |
| `<tag>_fit_summary.json` | Lattice parameters, twist (signed), errors, R², χ², settings |
| `<tag>_panelA_lattice_vs_position.csv` | Per-position *a*, *c*, errors, reliability flags |
| `<tag>_panelAB_lattice_mean_per_height.csv` | Mean *a*, *c* per height (panels a, b) |
| `<tag>_panelA_global_values.csv` | Global lattice values (median and alternatives) |
| `<tag>_panelF_histogram_per_height_vs_angle.csv` | Intensity vs angle for each height (panel f) |
| `<tag>_panelG_peak_angle_per_height.csv` | Peak angle, Gaussian width and errors, S/N, used-in-fit flag per height |
| `<tag>_panelF_intensity_vs_angle.csv` | Total (100) intensity vs angle |
| `<tag>_intensity_cube.csv` | (100) intensity for every angle, height and position |
| `<tag>_panelCDE_heatmap_*.csv` | Data behind panels c–e |

## Main settings

| Setting | Default | Meaning |
|---|---|---|
| `Z_FLIP` | `True` | First scanned row is at the far end of the spicule (matches the heat maps). Override per dataset with `"z_flip"`. |
| `PEAK_ERR_MODE` | `"residual"` | How the twist uncertainty is estimated (see Method). |
| `G_ERRBAR` | `"gauss"` | Error bars in panel g: Gaussian-fit error of each centre, or `"scatter"`. |
| `PEAK_MIN_SNR` / `PEAK_MIN_FRAC` | 5 / 0.05 | Minimum signal for a height to enter the twist fit. |
| `PEAK_WIN_DEG` | 0.8 | Half-width (deg) of the Gaussian fit window. |
| `PEAK_PROFILE` | `"sum"` | Per-height profile: sum across the spicule or brightest pixel. |
| `SHOW_EXCLUDED` | `False` | Also draw heights not used in the fit (grey). |
| `LAT_MEAN_ERR` | `"std"` | Error bars in panels a, b: SD or s.e.m. |
| `MOTOR_ERR_DEG` | 0 | Rotation-stage uncertainty (deg) added in quadrature, if known. |

## Notes and limitations

* Only heights whose (100) peak lies inside the scanned angular range can
  enter the twist fit. For strongly twisted samples, a wider rotation range
  (or a finer height step) gives more points.
* With a 5 µm height step the twist is resolved on that length scale only.
* The sign of the twist depends on sample mounting and rotation direction;
  compare signs only between datasets recorded with the same geometry.

## Citation

If you use this code, please cite the associated publication (to be added) and
this repository.

## License

MIT (see `LICENSE`).
