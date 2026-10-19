"""Surrogates for the exchange chemical potentials Δμ_i = μ_i − μ_ref at fixed temperature.

``fit_delta_mu`` fits a multivariate polynomial in the independent mole fractions x_j
(every element except the reference):

    Δμ_i(x) = intercept + Σ_k c_k Π_j x_j^power[k, j]

The ``feature_powers`` array gives the exponent of each variable in each term, so the
stored fit can be evaluated without scikit-learn, in any language. ``fit_delta_mu`` puts
the constant term among the features and stores the intercept as 0.
``build_spline_surrogate`` and ``build_gm_spline_surrogate`` build cubic-spline
surrogates for binary systems instead.

``save_fits`` writes the polynomial fits to HDF5 in this layout:

    /metadata                    attributes reference_element, x_variables (comma-
                                 separated), all_elements, poly_degree, source_db,
                                 fit_date (ISO 8601) and n_temperatures
    /T_{T_K:.2f}K/metadata       attributes temperature_K, temperature_C and
                                 temperature_units
    /T_{T_K:.2f}K/delta_mu_{EL}  datasets coefficients (float64, shape (n_features,))
                                 and feature_powers (int32, shape (n_features, n_vars));
                                 attributes intercept, rmse_J_per_mol, r2, n_points,
                                 reference_element, x_variables, description and
                                 delta_mu_formula
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from phasefoundry.thermodynamics.storage import load_table

_logger = logging.getLogger(__name__)


def fit_delta_mu(
    table_paths: list[str | Path],
    reference_element: str,
    poly_degree: int = 4,
    temperatures_K: list[float] | None = None,
) -> list[dict]:
    """Fit Δμ_i = μ_i − μ_ref as a polynomial in composition at each temperature.

    Every element except the reference gets its own fit. Points where Δμ_i is not finite
    are left out. One summary line per temperature is printed.

    Args:
        table_paths: HDF5 tables from ``generate_table``, one per temperature. Each must
            contain ``MU``.
        reference_element: Reference element of the exchange potential, e.g. ``"V"``.
        poly_degree: Polynomial degree.
        temperatures_K: Temperatures in K, one per table, used instead of the values
            stored in the tables.

    Returns:
        One dict per temperature with ``"temperature_K"``, ``"temperature_C"``,
        ``"elements"``, ``"x_variables"`` (the independent elements),
        ``"reference_element"``, ``"poly_degree"`` and ``"models"``. ``"models"`` maps each
        fitted element to ``"coefficients"`` (shape ``(n_features,)``), ``"intercept"``
        (0.0), ``"feature_powers"`` (shape ``(n_features, n_vars)``), ``"rmse"`` in J/mol,
        ``"r2"``, ``"n_points"`` and ``"description"``.

    Raises:
        ValueError: If ``reference_element`` is not in a table.
        KeyError: If a table has no ``MU`` data.
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

        X = data["compositions"][:, col_indices]
        mu_ref = data["MU"][reference_element]

        poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
        X_poly = poly.fit_transform(X)
        feature_powers = poly.powers_.astype(np.int32)

        models = {}
        fitted_elements = [el for el in elements if el != reference_element]

        for el in fitted_elements:
            delta_mu = data["MU"][el] - mu_ref

            valid = np.isfinite(delta_mu)
            reg = LinearRegression(fit_intercept=False)  # the bias column comes from PolynomialFeatures
            reg.fit(X_poly[valid], delta_mu[valid])

            y_pred = reg.predict(X_poly[valid])
            rmse = float(np.sqrt(mean_squared_error(delta_mu[valid], y_pred)))
            r2 = float(r2_score(delta_mu[valid], y_pred))

            coeffs = reg.coef_.copy()
            intercept = 0.0

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

        _logger.info(
            "%s",
            f"T = {T_K:.1f} K ({T_C:.1f} °C)  |  "
            + "  ".join((f"Δμ_{el}: RMSE={m['rmse']:.1f} J/mol  R²={m['r2']:.4f}" for el, m in models.items())),
        )

    return results


def build_spline_surrogate(
    table_paths: list[str | Path],
    reference_element: str,
    temperatures_K: list[float] | None = None,
) -> dict:
    """Build cubic-spline surrogates for Δμ, and its integral, at each temperature.

    Each Δμ_i is interpolated with a ``scipy.interpolate.CubicSpline`` through the
    pycalphad points, as a function of x_i alone, so the surrogate is meant for binary
    systems. The spline passes through every data point and has no polynomial (Runge)
    oscillations. Its antiderivative ∫Δμ dx gives G_m up to a constant. One summary line
    per temperature is printed.

    Args:
        table_paths: HDF5 tables from ``generate_table``, one per temperature. Each must
            contain ``MU``.
        reference_element: Reference element, e.g. ``"V"``.
        temperatures_K: Temperatures in K, one per table, used instead of the values
            stored in the tables.

    Returns:
        A dict with the same top-level keys as ``load_fits``: ``"temperatures_K"``
        (array), ``"metadata"`` and ``"fits"``. ``"fits"`` is keyed by temperature in K;
        each entry has ``"x_variables"``, ``"reference_element"`` and ``"models"``, which
        maps each independent element to ``"spline"`` (Δμ(x)), ``"spline_antideriv"``
        (∫Δμ dx), ``"x_data"``, ``"dmu_data"``, ``"rmse"`` (0.0) and ``"r2"`` (1.0).
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
            spl_intg = spl.antiderivative()

            models[el] = {
                "spline": spl,
                "spline_antideriv": spl_intg,
                "x_data": x,
                "dmu_data": dmu,
                "rmse": 0.0,  # the spline interpolates the data
                "r2": 1.0,
            }

        fits[T_K] = {
            "models": models,
            "x_variables": x_vars,
            "reference_element": reference_element,
        }

        _logger.info("T = %.1f K (%.1f °C)  |  spline built on %s pts", T_K, T_K - 273.15, valid.sum())

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
    """Build a cubic-spline surrogate of G_m itself at each temperature.

    The spline interpolates the pycalphad G_m points exactly, so the double-well shape has
    no integration error, and the phase-field driving force is its analytic derivative
    (``CubicSpline.derivative()``). Fitting Δμ and integrating, as
    ``build_spline_surrogate`` does, accumulates errors that can flatten a shallow double
    well and push the equilibrium compositions to 0 and 1.

    The composition variable is the mole fraction of the reference element, over the range
    present in the table, so the surrogate is meant for binary systems. One summary line
    per temperature is printed.

    Args:
        table_paths: HDF5 tables from ``generate_table``, one per temperature. Each must
            contain ``GM``.
        reference_element: Reference element, e.g. ``"V"``; its mole fraction is the
            composition axis.
        temperatures_K: Temperatures in K, one per table, used instead of the values
            stored in the tables.

    Returns:
        A dict with the same top-level keys as ``load_fits``: ``"temperatures_K"``
        (array), ``"metadata"`` and ``"fits"``. ``"fits"`` is keyed by temperature in K;
        each entry has ``"x_variables"``, ``"reference_element"`` and ``"models"``, whose
        one entry, keyed by the reference element, holds ``"spline_gm"`` (G_m(x_ref)),
        ``"spline_dfdx"`` (dG_m/dx_ref), ``"x_data"`` and ``"gm_data"`` (J/mol).

    Raises:
        KeyError: If a table has no ``GM`` data.
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
        spl_dfdx = spl_gm.derivative()

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

        _logger.info(
            "T = %.1f K (%.1f °C)  |  Gm spline built on %s pts  x_V=[%.4f, %.4f]",
            T_K,
            T_K - 273.15,
            valid.sum(),
            x_V[0],
            x_V[-1],
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
    """Save polynomial fits to HDF5, in the layout described in the module docstring.

    Args:
        fit_results: Output of ``fit_delta_mu``.
        output_path: Output file path. An existing file is overwritten and missing parent
            directories are created.
        source_db: Path of the TDB database the tables came from, stored as metadata.
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

    _logger.info("\nSaved fits → %s  (%s temperatures, reference = %s)", path, len(fit_results), ref0["reference_element"])


def load_fits(path: str | Path) -> dict:
    """Load polynomial fits written by ``save_fits``.

    Args:
        path: HDF5 file written by ``save_fits``.

    Returns:
        A dict with ``"metadata"`` (the file's global attributes), ``"temperatures_K"``
        (sorted list) and ``"fits"``, keyed by temperature in K. Each fit has
        ``"temperature_K"``, ``"temperature_C"`` and ``"models"``, which maps each fitted
        element to ``"coefficients"``, ``"feature_powers"``, ``"intercept"``, ``"rmse"``
        (J/mol), ``"r2"``, ``"n_points"``, ``"x_variables"`` and ``"description"``.
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
    """Evaluate the polynomial fits of Δμ_i at given compositions.

    The polynomial is rebuilt from ``coefficients`` and ``feature_powers`` with NumPy
    only, without scikit-learn.

    Args:
        fit: One temperature entry, ``load_fits(path)["fits"][T_K]``.
        compositions: Mole fractions of the independent elements, in the order of
            ``fit["models"][el]["x_variables"]``, shape ``(n_pts, n_vars)``.

    Returns:
        Δμ_i in J/mol for each fitted element, each of shape ``(n_pts,)``.
    """
    results = {}
    for el, m in fit["models"].items():
        coeffs = m["coefficients"]
        powers = m["feature_powers"]

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
    """Write the ``n_terms`` largest-magnitude terms of a fitted polynomial on one line."""
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
