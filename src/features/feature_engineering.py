"""
Feature engineering for astronomical source catalogues.

Adds derived features on top of the raw photometric/morphological columns:
  - Colour ratios and colour excess
  - Magnitude offsets
  - Morphology indices (compactness, elongation)
  - SNR-normalised residuals
  - Log-scaled features for heavy-tailed distributions

All transformations are deterministic (no fitting required) and safe to
apply identically to train and test splits.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "snr", "fwhm_arcsec", "ellipticity", "flux_radius",
    "class_star", "kron_radius", "mag_auto", "mag_err",
    "a_b_ratio", "g_r_color", "r_i_color", "background",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append derived features.

    Parameters
    ----------
    df : pd.DataFrame
        Catalogue with at least the columns in RAW_FEATURES.

    Returns
    -------
    pd.DataFrame with additional columns.
    """
    out = df.copy()

    # Log-scaled heavy-tailed features
    out["log_snr"]        = np.log1p(out["snr"].clip(lower=0))
    out["log_flux_radius"] = np.log1p(out["flux_radius"].clip(lower=0))
    out["log_kron_radius"] = np.log1p(out["kron_radius"].clip(lower=0))
    out["log_background"]  = np.log1p(out["background"].clip(lower=0))

    # Compactness: ratio of PSF-like size to source size
    eps = 1e-6
    out["compactness"]     = out["fwhm_arcsec"] / (out["flux_radius"] + eps)
    out["elongation"]      = out["a_b_ratio"].clip(lower=1.0)

    # Morphology-colour coupling
    out["size_color_ratio"] = out["flux_radius"] / (np.abs(out["g_r_color"]) + 0.1)

    # SNR-weighted magnitude uncertainty
    out["snr_mag_product"]  = out["log_snr"] * (1.0 - out["mag_err"].clip(0, 1))

    # Colour-colour index (stellar locus offset)
    out["cc_index"]         = out["g_r_color"] - 2.0 * out["r_i_color"]

    # Background-subtracted SNR proxy
    out["bg_snr_ratio"]     = out["log_snr"] / (out["log_background"] + eps)

    # Concentration: ratio of morphological radii
    out["concentration"]    = out["kron_radius"] / (out["flux_radius"] + eps)

    # Stellarity-morphology product
    out["stellarity_morph"] = out["class_star"] * (1.0 - out["ellipticity"])

    return out


def get_feature_columns(include_derived: bool = True) -> list[str]:
    """Return list of feature column names."""
    if not include_derived:
        return list(RAW_FEATURES)
    derived = [
        "log_snr", "log_flux_radius", "log_kron_radius", "log_background",
        "compactness", "elongation", "size_color_ratio", "snr_mag_product",
        "cc_index", "bg_snr_ratio", "concentration", "stellarity_morph",
    ]
    return list(RAW_FEATURES) + derived
