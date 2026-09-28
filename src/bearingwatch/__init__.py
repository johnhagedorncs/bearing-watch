"""bearing-watch: run-to-failure bearing anomaly detection.

Pipeline: ingest -> features -> train (healthy window only) -> score -> detect
-> serve -> monitor. Every training run writes one self-describing run folder;
inference loads a run folder by path.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
