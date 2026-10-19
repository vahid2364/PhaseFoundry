"""Polynomial fitting of exchange chemical potentials Δμ_i = μ_i - μ_ref
as a function of composition, at fixed temperature.

The fitted models are stored in HDF5 with enough information to fully
reconstruct the polynomial without sklearn — making them portable to
phase-field codes in any language.

Polynomial form (multivariate)
-------------------------------
    Δμ_i(x) = intercept + Σ_k  coeff_k · Π_j  x_j ^ power[k, j]

where x_j are the independent mole fractions (all elements except the
reference). The ``feature_powers`` dataset encodes the exponent of each
variable in each term, so the model can be reconstructed from scratch.

HDF5 layout
-----------
/metadata/
    reference_element        str   e.g. "V"
    x_variables              str   e.g. "NB" (binary) or "HF,NB,TI"
    poly_degree              int
    source_db                str
    fit_date                 str (ISO 8601)

/T_{T_K}K/
    /metadata/
        temperature_K        float
        temperature_C        float
    /delta_mu_{EL}/
        coefficients         float64  (n_features,)   — one per polynomial term
        intercept            float64  scalar
        feature_powers       int32    (n_features, n_vars) — exponent per term per variable
        x_variables          str      comma-separated variable names (redundant but handy)
        rmse                 float64  J/mol
        r2                   float64
        n_points             int
        description          str      human-readable equation summary
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from .storage import load_table


def fit_delta_mu(
    table_paths: list[str | Path],
    reference_element: str,
    poly_degree: int = 4,
    temperatures_K: list[float] | None = None,
) -> list[dict]:
    """Fit Δμ_i = μ_i − μ_ref as a polynomial in composition at each temperature.

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 tables (one per temperature) from generate_table().
        Must contain 'MU' datasets.
    reference_element : str
        Reference element for the exchange potential, e.g. 'V'.
    poly_degree : int
        Polynomial degree. Default 4.
    temperatures_K : list of float, optional
        Override temperatures (K). If None, read from file metadata.

    Returns:
    -------
    list of dict, one per temperature, each containing:
        'temperature_K'     float
        'temperature_C'     float
        'elements'          list[str]   all elements
        'x_variables'       list[str]   independent composition variables
        'reference_element' str
        'poly_degree'       int
        'models'            dict[str, dict]  one entry per fitted element:
            'coefficients'      np.ndarray  (n_features,)
            'intercept'         float
            'feature_powers'    np.ndarray  (n_features, n_vars)  int
            'rmse'              float   J/mol
            'r2'                float
            'n_points'          int
    """
    results = []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if reference_element not in elements:
            raise ValueError(f"reference_element '{reference_element}' not in elements {elements}")
        if "MU" not in data:
            raise KeyError(f"'MU' not stored in {path}. Re-generate with quantities=('MU',...)")

        T_K = temperatures_K[i] if temperatures_K is not None else float(data["metadata"]["temperature"])
        T_C = T_K - 273.15

        x_vars = [el for el in elements if el != reference_element]
        col_indices = [elements.index(el) for el in x_vars]

        X = data["compositions"][:, col_indices]  # (n_pts, n_vars)
        mu_ref = data["MU"][reference_element]  # (n_pts,)

        # Build polynomial features (shared transformer for all elements)
        poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
        X_poly = poly.fit_transform(X)
        feature_powers = poly.powers_.astype(np.int32)  # (n_features, n_vars)

        models = {}
        fitted_elements = [el for el in elements if el != reference_element]

        for el in fitted_elements:
            delta_mu = data["MU"][el] - mu_ref

            valid = np.isfinite(delta_mu)
            reg = LinearRegression(fit_intercept=False)  # bias included in poly features
            reg.fit(X_poly[valid], delta_mu[valid])

            y_pred = reg.predict(X_poly[valid])
            rmse = float(np.sqrt(mean_squared_error(delta_mu[valid], y_pred)))
            r2 = float(r2_score(delta_mu[valid], y_pred))

            # Coefficients align 1-to-1 with feature_powers rows
            coeffs = reg.coef_.copy()
            intercept = 0.0  # absorbed into bias term (first feature = 1^0)

            desc = _describe_poly(el, reference_element, x_vars, coeffs, feature_powers, poly_degree)

            models[el] = {
                "coefficients": coeffs,
                "intercept": intercept,
                "feature_powers": feature_powers,
                "rmse": rmse,
                "r2": r2,
                "n_points": int(valid.sum()),
                "description": desc,
            }

        results.append(
            {
                "temperature_K": T_K,
                "temperature_C": T_C,
                "elements": elements,
                "x_variables": x_vars,
                "reference_element": reference_element,
                "poly_degree": poly_degree,
                "models": models,
            }
        )

        print(
            f"T = {T_K:.1f} K ({T_C:.1f} °C)  |  "
            + "  ".join(f"Δμ_{el}: RMSE={m['rmse']:.1f} J/mol  R²={m['r2']:.4f}" for el, m in models.items())
        )

    return results


def build_spline_surrogate(
    table_paths: list[str | Path],
    reference_element: str,
    temperatures_K: list[float] | None = None,
) -> dict:
    """Build a CubicSpline surrogate for Δμ and G_m at each temperature.

    Instead of fitting a global polynomial, this constructs a
    ``scipy.interpolate.CubicSpline`` directly through the pycalphad data
    points.  The spline passes exactly through every data point and is
    C² continuous, giving near-perfect fit quality (R² ≈ 1.0) with no
    Runge oscillations.

    G_m is obtained by analytic integration of the cubic spline for Δμ
    (each piecewise cubic integrates to a quartic, computed exactly).

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 tables (one per temperature) from generate_table().
        Must contain 'MU' datasets.
    reference_element : str
        e.g. 'V'.
    temperatures_K : list of float, optional

    Returns:
    -------
    dict  — same top-level structure as ``load_fits()`` so it can be
    passed directly to the PFM solver:
        'temperatures_K'   np.ndarray
        'metadata'         dict
        'fits'             dict[float, dict]  keyed by temperature_K
            Each entry has:
            'models'  dict[str, dict]
                'spline'        CubicSpline   Δμ(x)
                'spline_antideriv'  PPoly     G_m(x) = ∫Δμ dx
                'x_data'        np.ndarray   composition grid
                'dmu_data'      np.ndarray   Δμ values
                'rmse'          float   (0 — interpolating spline)
                'r2'            float   (1.0 — interpolating spline)
    """
    from scipy.interpolate import CubicSpline

    fits = {}
    stored_T = []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        T_K = temperatures_K[i] if temperatures_K is not None else float(data["metadata"]["temperature"])
        stored_T.append(T_K)

        col_indices = {el: elements.index(el) for el in elements}
        mu_ref = data["MU"][reference_element]
        x_vars = [el for el in elements if el != reference_element]

        models = {}
        for el in x_vars:
            col = col_indices[el]
            x = data["compositions"][:, col]
            dmu = data["MU"][el] - mu_ref

            valid = np.isfinite(dmu)
            x, dmu = x[valid], dmu[valid]
            order = np.argsort(x)
            x, dmu = x[order], dmu[order]

            spl = CubicSpline(x, dmu, extrapolate=True)
            spl_intg = spl.antiderivative()  # PPoly — analytic ∫Δμ dx

            models[el] = {
                "spline": spl,
                "spline_antideriv": spl_intg,
                "x_data": x,
                "dmu_data": dmu,
                "rmse": 0.0,  # interpolating — exact
                "r2": 1.0,
            }

        fits[T_K] = {
            "models": models,
            "x_variables": x_vars,
            "reference_element": reference_element,
        }

        print(f"T = {T_K:.1f} K ({T_K - 273.15:.1f} °C)  |  spline built on {valid.sum()} pts")

    return {
        "temperatures_K": np.array(stored_T),
        "metadata": {
            "reference_element": reference_element,
            "x_variables": x_vars,
            "poly_degree": "spline",
            "surrogate_type": "CubicSpline",
        },
        "fits": fits,
    }


def build_gm_spline_surrogate(
    table_paths: list[str | Path],
    reference_element: str,
    temperatures_K: list[float] | None = None,
) -> dict:
    """Build a CubicSpline surrogate fitted **directly to G_m** at each temperature.

    This is the preferred approach for CALPHAD-PFM coupling because:

    * The spline interpolates the pycalphad G_m data exactly — the double-well
      shape is preserved with no integration error.
    * The PFM driving force df/dc is the **analytic derivative** of the spline
      (``CubicSpline.derivative()``), so no numerical differentiation is needed.
    * Fitting Δμ and integrating (the previous approach) accumulates errors that
      can destroy the shallow double-well and push equilibrium compositions to
      0 and 1.

    The composition variable is ``x_V`` (mol fraction of the reference element).
    The spline is built on x_V ∈ [x_min, x_max] as found in the table.

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 tables (one per temperature) from generate_table().
        Must contain 'GM' and 'MU' datasets.
    reference_element : str
        e.g. 'V'.  The composition axis is x_{reference_element}.
    temperatures_K : list of float, optional

    Returns:
    -------
    dict  — same top-level structure as ``load_fits()`` / ``build_spline_surrogate()``:
        'temperatures_K'   np.ndarray
        'metadata'         dict
        'fits'             dict[float, dict]
            Each entry has:
            'models'  dict[str, dict]  (keyed by reference element)
                'spline_gm'     CubicSpline   G_m(x_V)
                'spline_dfdx'   CubicSpline   dG_m/dx_V  (analytic derivative)
                'x_data'        np.ndarray    x_V composition grid
                'gm_data'       np.ndarray    G_m values  [J/mol]
    """
    from scipy.interpolate import CubicSpline

    fits = {}
    stored_T = []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if "GM" not in data:
            raise KeyError(f"'GM' not stored in {path}. Re-generate with quantities=('GM', ...).")

        T_K = temperatures_K[i] if temperatures_K is not None else float(data["metadata"]["temperature"])
        stored_T.append(T_K)

        ref_col = elements.index(reference_element)
        x_V = data["compositions"][:, ref_col]
        GM = data["GM"]

        valid = np.isfinite(GM)
        x_V, GM = x_V[valid], GM[valid]
        order = np.argsort(x_V)
        x_V, GM = x_V[order], GM[order]

        spl_gm = CubicSpline(x_V, GM, extrapolate=True)
        spl_dfdx = spl_gm.derivative()  # dG_m/dx_V  (analytic)

        fits[T_K] = {
            "models": {
                reference_element: {
                    "spline_gm": spl_gm,
                    "spline_dfdx": spl_dfdx,
                    "x_data": x_V,
                    "gm_data": GM,
                }
            },
            "x_variables": [reference_element],
            "reference_element": reference_element,
        }

        print(
            f"T = {T_K:.1f} K ({T_K - 273.15:.1f} °C)  |  Gm spline built on {valid.sum()} pts  x_V=[{x_V[0]:.4f}, {x_V[-1]:.4f}]"
        )

    x_vars = [reference_element]
    return {
        "temperatures_K": np.array(stored_T),
        "metadata": {
            "reference_element": reference_element,
            "x_variables": x_vars,
            "poly_degree": "gm_spline",
            "surrogate_type": "CubicSpline_on_Gm",
        },
        "fits": fits,
    }


def save_fits(
    fit_results: list[dict],
    output_path: str | Path,
    source_db: str = "",
) -> None:
    """Save polynomial fit results to HDF5.

    Parameters
    ----------
    fit_results : list of dict
        Output of fit_delta_mu().
    output_path : str or Path
        Output HDF5 file path.
    source_db : str
        Path to the original TDB file (stored as metadata).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref0 = fit_results[0]

    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["reference_element"] = ref0["reference_element"]
        meta.attrs["x_variables"] = ",".join(ref0["x_variables"])
        meta.attrs["all_elements"] = ",".join(ref0["elements"])
        meta.attrs["poly_degree"] = ref0["poly_degree"]
        meta.attrs["source_db"] = str(source_db)
        meta.attrs["fit_date"] = datetime.now(UTC).isoformat()
        meta.attrs["n_temperatures"] = len(fit_results)

        for res in fit_results:
            T_K = res["temperature_K"]
            T_C = res["temperature_C"]
            grp = f.create_group(f"T_{T_K:.2f}K")

            tm = grp.create_group("metadata")
            tm.attrs["temperature_K"] = T_K
            tm.attrs["temperature_C"] = T_C
            tm.attrs["temperature_units"] = "K (stored); °C shown for reference"

            for el, m in res["models"].items():
                eg = grp.create_group(f"delta_mu_{el}")

                ds_c = eg.create_dataset("coefficients", data=m["coefficients"])
                ds_c.attrs["units"] = "J/mol"
                ds_c.attrs["description"] = "Coefficients c_k such that Δμ = Σ_k c_k · Π_j x_j^power[k,j]"

                ds_p = eg.create_dataset("feature_powers", data=m["feature_powers"])
                ds_p.attrs["columns"] = ",".join(res["x_variables"])
                ds_p.attrs["description"] = "Row k: exponent of each composition variable in term k"

                eg.attrs["intercept"] = m["intercept"]
                eg.attrs["rmse_J_per_mol"] = m["rmse"]
                eg.attrs["r2"] = m["r2"]
                eg.attrs["n_points"] = m["n_points"]
                eg.attrs["reference_element"] = res["reference_element"]
                eg.attrs["x_variables"] = ",".join(res["x_variables"])
                eg.attrs["description"] = m["description"]
                eg.attrs["delta_mu_formula"] = f"Δμ_{el} = μ_{el} - μ_{res['reference_element']}  [J/mol]"

    print(f"\nSaved fits → {path}  ({len(fit_results)} temperatures, reference = {ref0['reference_element']})")


def load_fits(path: str | Path) -> dict:
    """Load polynomial fit results from HDF5.

    Returns:
    -------
    dict with keys:
        'metadata'        dict  (global metadata)
        'temperatures_K'  list[float]
        'fits'            dict[float, dict]  keyed by T_K:
            'temperature_K', 'temperature_C',
            'models': dict[str, dict] with keys:
                'coefficients', 'feature_powers', 'intercept', 'rmse', 'r2'
    """
    path = Path(path)
    out = {"metadata": {}, "temperatures_K": [], "fits": {}}

    with h5py.File(path, "r") as f:
        out["metadata"] = dict(f["metadata"].attrs)

        for key in f:
            if not key.startswith("T_"):
                continue
            grp = f[key]
            T_K = float(grp["metadata"].attrs["temperature_K"])
            T_C = float(grp["metadata"].attrs["temperature_C"])
            out["temperatures_K"].append(T_K)

            models = {}
            for el_key in grp:
                if not el_key.startswith("delta_mu_"):
                    continue
                el = el_key[len("delta_mu_") :]
                eg = grp[el_key]
                models[el] = {
                    "coefficients": eg["coefficients"][:],
                    "feature_powers": eg["feature_powers"][:],
                    "intercept": float(eg.attrs["intercept"]),
                    "rmse": float(eg.attrs["rmse_J_per_mol"]),
                    "r2": float(eg.attrs["r2"]),
                    "n_points": int(eg.attrs["n_points"]),
                    "x_variables": eg.attrs["x_variables"].split(","),
                    "description": eg.attrs.get("description", ""),
                }
            out["fits"][T_K] = {
                "temperature_K": T_K,
                "temperature_C": T_C,
                "models": models,
            }

    out["temperatures_K"].sort()
    return out


def eval_delta_mu(
    fit: dict,
    compositions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate Δμ_i at given compositions using stored polynomial coefficients.

    This function does NOT require sklearn — it reconstructs the polynomial
    from ``coefficients`` and ``feature_powers`` directly.

    Parameters
    ----------
    fit : dict
        A single temperature entry from load_fits()['fits'][T_K].
    compositions : np.ndarray, shape (n_pts, n_vars)
        Mole fractions of the independent elements in the same order as
        fit['models'][el]['x_variables'].

    Returns:
    -------
    dict[str, np.ndarray]  Δμ_i in J/mol for each fitted element.
    """
    results = {}
    for el, m in fit["models"].items():
        coeffs = m["coefficients"]  # (n_features,)
        powers = m["feature_powers"]  # (n_features, n_vars)

        # Compute each polynomial term: Π_j x_j^power[k,j]
        n_pts = compositions.shape[0]
        n_features = len(coeffs)
        terms = np.ones((n_pts, n_features))
        for k in range(n_features):
            for j, exp in enumerate(powers[k]):
                if exp > 0:
                    terms[:, k] *= compositions[:, j] ** exp

        results[el] = terms @ coeffs + m["intercept"]
    return results


def _describe_poly(
    el: str,
    ref: str,
    x_vars: list[str],
    coeffs: np.ndarray,
    powers: np.ndarray,
    degree: int,
    n_terms: int = 5,
) -> str:
    """Build a short human-readable polynomial description."""
    # Sort by absolute coefficient magnitude, show top n_terms
    order = np.argsort(-np.abs(coeffs))
    terms = []
    for k in order[:n_terms]:
        c = coeffs[k]
        pw = powers[k]
        var_str = (
            "·".join((f"x_{x_vars[j]}^{pw[j]}" if pw[j] > 1 else f"x_{x_vars[j]}") for j in range(len(x_vars)) if pw[j] > 0)
            or "1"
        )
        terms.append(f"{c:+.2e}·{var_str}")
    return f"Δμ_{el} - μ_{ref}  ≈  " + " ".join(terms) + "  + ..."
