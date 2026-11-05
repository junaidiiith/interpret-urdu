"""interpret-urdu: mechanistic interpretability toolkit for the CHR Urdu study.

Public entry points used by the paper's RQ1–RQ3 pipelines and by
``examples/quickstart_*.py``.
"""

from urdu_interp import (
    alignment,
    causal,
    concept_attribution,
    convergence,
    corpora,
    extraction,
    finetune,
    models,
    patchscopes,
    pipeline,
    script_utils,
    stats,
    tasks,
)

__all__ = [
    "alignment",
    "causal",
    "concept_attribution",
    "convergence",
    "corpora",
    "extraction",
    "finetune",
    "models",
    "patchscopes",
    "pipeline",
    "script_utils",
    "stats",
    "tasks",
]

__version__ = "0.1.0"
