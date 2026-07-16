"""
Plotting utilities for thermodynamic tables.

Main entry point: plot_free_energy_vs_composition()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .storage import load_table


def plot_free_energy_vs_composition(
    table_paths: list[str | Path],
    x_element: str,
    quantity: str = "GM",
    temperatures_celsius: list[float] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a thermodynamic quantity vs mole fraction of one element
    for multiple temperatures (one curve per table file).

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 table files produced by generate_table(), one per temperature.
        They are plotted in the order given (typically sorted by T).
    x_element : str
        Element whose mole fraction goes on the x-axis, e.g. 'V'.
    quantity : str
        Which quantity to plot: 'GM' (default) or a chemical potential
        string like 'MU/HF'.  For 'GM' pass 'GM'; for a chemical
        potential pass 'MU' and the element via ``mu_element``.
    temperatures_celsius : list of float, optional
        Labels for each curve.  If None, the temperature is read from
        each file's metadata (stored in K, converted to °C for display).
    ax : matplotlib Axes, optional
        Axes to draw on.  A new figure is created when None.
    figsize : tuple
        Figure size when a new figure is created.
    save_path : str or Path, optional
        If given, save the figure to this path.
    show : bool
        Call plt.show() at the end.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.plasma
    n = len(table_paths)

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if x_element not in elements:
            raise ValueError(
                f"'{x_element}' not in elements {elements} for file {path}"
            )

        col_idx = elements.index(x_element)
        x = data["compositions"][:, col_idx]

        if quantity == "GM":
            if "GM" not in data:
                raise KeyError(f"'GM' not stored in {path}. Re-generate with quantities=('GM',...)")
            y = data["GM"]
            ylabel = "Molar Gibbs energy  $G_m$  [J/mol]"
        elif quantity.startswith("MU/"):
            el = quantity.split("/", 1)[1]
            if "MU" not in data or el not in data["MU"]:
                raise KeyError(f"MU/{el} not stored in {path}")
            y = data["MU"][el]
            ylabel = f"Chemical potential  $\\mu_{{\\rm {el.title()}}}$  [J/mol]"
        else:
            raise ValueError(f"Unknown quantity '{quantity}'. Use 'GM' or 'MU/<element>'.")

        # Sort by x for a clean line
        order = np.argsort(x)
        x_sorted, y_sorted = x[order], y[order]

        # Temperature label
        if temperatures_celsius is not None:
            label = f"{temperatures_celsius[i]:.0f} °C"
        else:
            T_K = data["metadata"].get("temperature", 0)
            label = f"{T_K - 273.15:.0f} °C"

        color = cmap(i / max(n - 1, 1))
        ax.plot(x_sorted, y_sorted, color=color, label=label, linewidth=1.8)

    ax.set_xlabel(f"$x_{{\\rm {x_element.title()}}}$  (mol fraction)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if temperatures_celsius is not None and len(temperatures_celsius) == 1:
        title = f"{quantity} vs $x_{{\\rm {x_element.title()}}}$ at {temperatures_celsius[0]:.0f} °C"
    else:
        title = f"{quantity} vs $x_{{\\rm {x_element.title()}}}$"
    ax.set_title(title, fontsize=13)
    ax.legend(
        title="Temperature", fontsize=9, title_fontsize=9,
        loc="best", ncol=2 if n > 6 else 1,
    )
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return ax


def plot_chemical_potential_vs_composition(
    table_paths: list[str | Path],
    x_element: str,
    mu_elements: list[str] | None = None,
    temperatures_celsius: list[float] | None = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[plt.Axes]:
    """
    Plot chemical potentials μᵢ vs mole fraction for multiple temperatures.
    One subplot per element in mu_elements.
    """
    sample = load_table(table_paths[0])
    elements = sample["elements"]

    if mu_elements is None:
        mu_elements = elements  # all elements

    fig, axes = plt.subplots(1, len(mu_elements), figsize=figsize, sharey=False)
    if len(mu_elements) == 1:
        axes = [axes]

    for j, el in enumerate(mu_elements):
        plot_free_energy_vs_composition(
            table_paths=table_paths,
            x_element=x_element,
            quantity=f"MU/{el}",
            temperatures_celsius=temperatures_celsius,
            ax=axes[j],
            show=False,
        )
        axes[j].set_title(f"$\\mu_{{\\rm {el.title()}}}$", fontsize=12)
        if j > 0:
            axes[j].set_ylabel("")

    plt.suptitle(
        f"Chemical potentials vs $x_{{\\rm {x_element.title()}}}$",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return axes
