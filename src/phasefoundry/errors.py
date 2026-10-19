"""Exceptions raised by PhaseFoundry."""

from __future__ import annotations


class ConfigError(ValueError):
    """Signal that a config file or ``--set`` override has an invalid key or value.

    It is raised for missing sections or keys, unknown option values, and settings that
    contradict each other.
    """


class MissingDependencyError(ImportError):
    """Signal that an optional package needed for the requested feature is not installed.

    The message names the package and the extra that installs it.
    """
