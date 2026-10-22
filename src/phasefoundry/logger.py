"""
phasefoundry.logger — Structured logging and timing for the pipeline
====================================================================

Provides:
  - Colored console output + rotating file log (run.log)
  - Structured timing records written to timings.jsonl (one JSON line per step)
  - Config fingerprinting (SHA-256 of config + TDB) to link outputs to inputs
  - Per-run memory tracking (peak RAM / VRAM)
  - Free-energy monotonicity check (warns on numerical instability)
  - Step-rate histogram (steps/s over simulation time)
  - Pipeline summary table printed after pf-run-all

Usage
-----
In any workflow script::

    from phasefoundry.logger import get_logger

    log = get_logger(cfg, step="simulate", results_dir=results_dir)
    log.info("Starting simulation")

    with log.timing("simulation"):
        result = solve_cahn_hilliard(...)

    log.memory()
    log.check_monotonicity(result.free_energy_history)
    log.step_rate_histogram(result.step_times, result.elapsed_seconds,
                            output_dir=output_dir)

Disable entirely by omitting the ``logging`` block in the YAML config (default)
or setting ``logging.enabled: false``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import colorama
    colorama.init(autoreset=True)
    _COLORAMA = True
except ImportError:
    _COLORAMA = False

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import cupy as _cp
    _CUPY = True
except ImportError:
    _CUPY = False


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_GRAY   = "\033[90m"

def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes (no-op if stdout is not a tty)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


# ── default logging config ────────────────────────────────────────────────────

_LOGGING_DEFAULTS: dict[str, Any] = {
    "enabled":           False,
    "level":             "info",
    "log_file":          "auto",
    "timings_file":      "auto",
    "progress_interval": 1000,
    "track_memory":      True,
    "fingerprint":       True,
    "pipeline_summary":  True,
}


def get_logging_cfg(cfg: dict) -> dict:
    """Return the effective logging sub-config with defaults applied."""
    raw = cfg.get("logging", {}) or {}
    out = dict(_LOGGING_DEFAULTS)
    out.update(raw)
    return out


# ── NullLogger (no-op) ───────────────────────────────────────────────────────

class NullLogger:
    """Drop-in replacement when logging is disabled — all calls are no-ops."""

    def info(self, *a, **kw):          pass
    def debug(self, *a, **kw):         pass
    def warning(self, *a, **kw):       pass
    def error(self, *a, **kw):         pass
    def section(self, *a, **kw):       pass
    def banner(self, *a, **kw):        pass
    def memory(self, *a, **kw):        pass
    def check_monotonicity(self, *a, **kw): pass
    def step_rate_histogram(self, *a, **kw): pass
    def log_timing_record(self, *a, **kw): pass
    def fingerprint(self, *a, **kw) -> str: return ""

    @contextmanager
    def timing(self, label: str = ""):
        t0 = time.perf_counter()
        yield
        self._last_elapsed = time.perf_counter() - t0

    @property
    def enabled(self) -> bool:
        return False


# ── PFLogger ──────────────────────────────────────────────────────────────────

class PFLogger:
    """
    Structured logger for a single pipeline step.

    Parameters
    ----------
    step_name : str
        Label shown in every log line, e.g. ``"simulate"``.
    results_dir : Path or None
        If given, writes ``run.log`` and ``timings.jsonl`` there.
    level : str
        ``"debug"`` | ``"info"`` | ``"warning"``
    log_file : str or Path
        ``"auto"`` → ``results_dir/run.log``; explicit path; or ``None`` to
        disable file output.
    timings_file : str or Path
        ``"auto"`` → ``results_dir/timings.jsonl``; or explicit path.
    track_memory : bool
        Whether to collect peak memory at end of timed sections.
    """

    def __init__(
        self,
        step_name:    str,
        results_dir:  Path | None = None,
        level:        str         = "info",
        log_file:     str | Path  = "auto",
        timings_file: str | Path  = "auto",
        track_memory: bool        = True,
    ):
        self.step_name    = step_name
        self.results_dir  = Path(results_dir) if results_dir else None
        self.track_memory = track_memory
        self._timings: list[dict] = []
        self._last_elapsed: float = 0.0

        # ── Python logging (file handler) ─────────────────────────────────────
        lvl_map = {"debug": logging.DEBUG, "info": logging.INFO,
                   "warning": logging.WARNING}
        log_level = lvl_map.get(level.lower(), logging.INFO)

        self._logger = logging.getLogger(f"phasefoundry.{step_name}")
        self._logger.setLevel(log_level)
        self._logger.propagate = False
        self._logger.handlers.clear()

        # file handler
        self._log_path: Path | None = None
        if log_file is not None and results_dir is not None:
            self._log_path = (
                Path(results_dir) / "run.log"
                if log_file == "auto"
                else Path(log_file)
            )
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self._log_path, mode="a", encoding="utf-8")
            fh.setLevel(log_level)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
            self._logger.addHandler(fh)

        # timings file
        self._timings_path: Path | None = None
        if timings_file is not None and results_dir is not None:
            self._timings_path = (
                Path(results_dir) / "timings.jsonl"
                if timings_file == "auto"
                else Path(timings_file)
            )
            self._timings_path.parent.mkdir(parents=True, exist_ok=True)

        self._session_start = datetime.now(timezone.utc).isoformat()

    # ── logging primitives ────────────────────────────────────────────────────

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _print(self, msg: str, color: str = "") -> None:
        line = f"{_c(self._ts(), _GRAY)}  {msg}"
        print(line)
        self._logger.info(msg)

    def info(self, msg: str) -> None:
        self._print(f"  {msg}")

    def debug(self, msg: str) -> None:
        if self._logger.isEnabledFor(logging.DEBUG):
            self._print(_c(f"  [debug] {msg}", _GRAY))

    def warning(self, msg: str) -> None:
        self._print(_c(f"  ⚠  {msg}", _YELLOW))
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._print(_c(f"  ✗  {msg}", _RED))
        self._logger.error(msg)

    def section(self, title: str) -> None:
        line = f"── {title} " + "─" * max(0, 54 - len(title))
        self._print(_c(line, _CYAN, _BOLD))

    def banner(self, title: str, config_path: str = "", fingerprint: str = "") -> None:
        width = 62
        print()
        print(_c("─" * width, _CYAN))
        print(_c(f"  {title}", _CYAN, _BOLD))
        if config_path:
            self._print(f"  Config     : {config_path}")
        if fingerprint:
            self._print(f"  Fingerprint: {_c(fingerprint, _BLUE)}")
        print(_c("─" * width, _CYAN))

    # ── timing context manager ────────────────────────────────────────────────

    @contextmanager
    def timing(self, label: str = ""):
        label = label or self.step_name
        t0 = time.perf_counter()
        peak_before = _peak_memory_mb() if self.track_memory else None
        self._logger.info(f"START {label}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._last_elapsed = elapsed
            peak_after = _peak_memory_mb() if self.track_memory else None
            mem_delta = (
                peak_after - peak_before
                if peak_before is not None and peak_after is not None
                else None
            )
            self._print(
                _c(f"  ✓  {label}  │  elapsed: {_fmt_elapsed(elapsed)}", _GREEN)
                + (f"  │  peak RAM: {peak_after:.0f} MB" if peak_after else "")
            )
            self._logger.info(
                f"END {label}  elapsed={elapsed:.3f}s"
                + (f"  peak_ram_mb={peak_after:.1f}" if peak_after else "")
            )
            record: dict[str, Any] = {
                "step":        self.step_name,
                "label":       label,
                "elapsed_s":   round(elapsed, 3),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }
            if peak_after is not None:
                record["peak_ram_mb"] = round(peak_after, 1)
            if mem_delta is not None:
                record["delta_ram_mb"] = round(mem_delta, 1)
            self._write_timing(record)

    # ── memory report ─────────────────────────────────────────────────────────

    def memory(self) -> dict:
        """Print and return current memory usage."""
        info: dict[str, float] = {}
        if _PSUTIL:
            proc = _psutil.Process()
            rss_mb = proc.memory_info().rss / 1e6
            info["ram_mb"] = round(rss_mb, 1)
            self._print(f"  Memory  RAM: {rss_mb:.0f} MB")
        if _CUPY:
            try:
                mem = _cp.cuda.Device().mem_info
                vram_used = (mem[1] - mem[0]) / 1e6
                info["vram_mb"] = round(vram_used, 1)
                self._print(f"          VRAM: {vram_used:.0f} MB")
            except Exception:
                pass
        return info

    # ── config fingerprinting ─────────────────────────────────────────────────

    def fingerprint(self, cfg: dict, tdb_path: str | Path | None = None) -> str:
        """
        Compute a short SHA-256 fingerprint of (config + TDB content).

        Returns a 12-character hex string, e.g. ``"a3f9c1b72e4d"``.
        Stored in timings.jsonl and printed in the banner.
        """
        h = hashlib.sha256()
        h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
        if tdb_path is not None:
            p = Path(tdb_path)
            if p.exists():
                h.update(p.read_bytes())
        digest = h.hexdigest()[:12]
        record = {
            "step":        self.step_name,
            "label":       "fingerprint",
            "fingerprint": digest,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        self._write_timing(record)
        return digest

    # ── free-energy monotonicity check ────────────────────────────────────────

    def check_monotonicity(
        self,
        F_hist: list[float] | np.ndarray,
        rise_tol_frac: float = 0.01,
        max_rise_frac: float = 0.10,
    ) -> bool:
        """
        Warn if free energy increases significantly during the simulation.

        Parameters
        ----------
        F_hist : array-like
            Free energy at each saved snapshot.
        rise_tol_frac : float
            A rise is flagged if ``ΔF > rise_tol_frac * |F_range|``.
        max_rise_frac : float
            Warn if more than this fraction of consecutive pairs show a rise.

        Returns
        -------
        bool
            True if free energy is well-behaved (no warning).
        """
        F = np.asarray(F_hist, dtype=float)
        if len(F) < 3:
            return True
        F_range = float(np.nanmax(F) - np.nanmin(F))
        tol = rise_tol_frac * F_range if F_range > 0 else 1.0
        diffs = np.diff(F)
        n_rises = int(np.sum(diffs > tol))
        frac    = n_rises / max(1, len(diffs))
        if frac > max_rise_frac:
            self.warning(
                f"Free energy increased in {n_rises}/{len(diffs)} intervals "
                f"({100*frac:.0f}%) — possible numerical instability. "
                "Consider reducing dt or increasing A_stab."
            )
            return False
        self.info(
            f"Free-energy check: {_c('OK', _GREEN)}  "
            f"({n_rises} minor rise(s), {100*frac:.1f}% of intervals)"
        )
        return True

    # ── step-rate histogram ───────────────────────────────────────────────────

    def step_rate_histogram(
        self,
        step_times: list[float] | np.ndarray,
        elapsed_total: float,
        output_dir: Path,
        n_bins: int = 30,
    ) -> Path | None:
        """
        Save a PNG of steps/s vs. simulation step index.

        Parameters
        ----------
        step_times : array-like
            Wall-clock timestamps (seconds since run start) at each saved
            snapshot, length = n_snapshots.
        elapsed_total : float
            Total elapsed wall-clock seconds (for the title).
        output_dir : Path
            Where to save ``step_rate.png``.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = np.asarray(step_times, dtype=float)
        if len(times) < 2:
            return None

        dt_wall = np.diff(times)
        dt_wall = np.where(dt_wall <= 0, np.nan, dt_wall)

        # Each interval covers save_every CH steps — infer from spacing
        # (just plot relative rate normalized to mean)
        rate = 1.0 / dt_wall                # snapshots per wall-second
        midpoints = 0.5 * (times[:-1] + times[1:])

        # rolling window smoothing (window = min 5, max len//4)
        win = max(1, min(5, len(rate) // 4))
        if win > 1:
            kernel = np.ones(win) / win
            rate_smooth = np.convolve(rate, kernel, mode="same")
        else:
            rate_smooth = rate

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(midpoints, rate, color="#b0c4de", linewidth=0.8, alpha=0.6,
                label="raw")
        ax.plot(midpoints, rate_smooth, color="#1f77b4", linewidth=1.6,
                label=f"smoothed (w={win})")
        ax.set_xlabel("Wall time [s]", fontsize=10)
        ax.set_ylabel("Snapshot rate [s⁻¹]", fontsize=10)
        ax.set_title(
            f"Simulation throughput  —  total: {_fmt_elapsed(elapsed_total)}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        out = Path(output_dir) / "step_rate.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.info(f"Step-rate histogram → {out.name}")
        return out

    # ── pipeline summary ──────────────────────────────────────────────────────

    @staticmethod
    def pipeline_summary(records: list[dict]) -> str:
        """
        Format a pipeline summary table from a list of timing records.

        Each record must have ``step``, ``elapsed_s``, and optionally
        ``status`` (``"ok"`` | ``"failed"`` | ``"skipped"``).
        """
        if not records:
            return ""
        col_w = max(len(r.get("label", r.get("step", ""))) for r in records) + 2
        lines = [
            "",
            _c("Pipeline summary", _CYAN, _BOLD),
            _c(f"{'Step':<{col_w}}  {'Elapsed':>10}  Status", _BOLD),
            _c("─" * (col_w + 22), _GRAY),
        ]
        total = 0.0
        for r in records:
            label   = r.get("label", r.get("step", "?"))
            elapsed = r.get("elapsed_s", 0.0)
            status  = r.get("status", "ok")
            total  += elapsed
            if status == "ok":
                sym = _c("✓", _GREEN)
            elif status == "failed":
                sym = _c("✗", _RED)
            else:
                sym = _c("–", _GRAY)
            lines.append(
                f"  {label:<{col_w}}  {_fmt_elapsed(elapsed):>10}  {sym}"
            )
        lines += [
            _c("─" * (col_w + 22), _GRAY),
            f"  {'Total':<{col_w}}  {_fmt_elapsed(total):>10}",
            "",
        ]
        return "\n".join(lines)

    # ── timings.jsonl writer ──────────────────────────────────────────────────

    def _write_timing(self, record: dict) -> None:
        self._timings.append(record)
        if self._timings_path is not None:
            with open(self._timings_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def log_timing_record(self, record: dict) -> None:
        """Write an arbitrary record to timings.jsonl (e.g. validate accuracy)."""
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        record.setdefault("step", self.step_name)
        self._write_timing(record)

    @property
    def enabled(self) -> bool:
        return True


# ── memory helpers ────────────────────────────────────────────────────────────

def _peak_memory_mb() -> float | None:
    if not _PSUTIL:
        return None
    try:
        return _psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


# ── factory ───────────────────────────────────────────────────────────────────

def get_logger(
    cfg:         dict,
    step:        str,
    results_dir: Path | None = None,
) -> PFLogger | NullLogger:
    """
    Return a PFLogger if ``logging.enabled: true`` in *cfg*, else NullLogger.

    Parameters
    ----------
    cfg : dict
        Full pipeline config dict.
    step : str
        Name of this pipeline step, e.g. ``"simulate"``.
    results_dir : Path or None
        Directory where ``run.log`` and ``timings.jsonl`` are written.
        Falls back to ``cfg['output']['results_dir']`` when None.
    """
    lcfg = get_logging_cfg(cfg)
    if not lcfg.get("enabled", False):
        return NullLogger()

    if results_dir is None:
        results_dir = cfg.get("output", {}).get("results_dir")

    return PFLogger(
        step_name    = step,
        results_dir  = Path(results_dir) if results_dir else None,
        level        = str(lcfg.get("level", "info")),
        log_file     = lcfg.get("log_file",     "auto"),
        timings_file = lcfg.get("timings_file", "auto"),
        track_memory = bool(lcfg.get("track_memory", True)),
    )


# ── read timings.jsonl helper (for pipeline summary) ─────────────────────────

def read_timings(results_dir: Path) -> list[dict]:
    """Read all records from timings.jsonl in results_dir."""
    p = Path(results_dir) / "timings.jsonl"
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
