"""Typed exceptions so every failure has a clear, user-readable cause."""


class SrfCoatingError(Exception):
    """Base class for all expected pipeline failures."""


class ConfigError(SrfCoatingError):
    """The YAML config is missing, malformed, or fails validation."""


class CstConnectionError(SrfCoatingError):
    """No usable CST Studio session or project could be attached."""


class CstExtractionError(SrfCoatingError):
    """CST is reachable but a requested result (mode, current, Q) is missing."""


class MeshParseError(SrfCoatingError):
    """The 3d.slim binary mesh file is missing or has an unexpected layout."""


class PhysicsError(SrfCoatingError):
    """A physical sanity check failed (weights, areas, currents, power)."""