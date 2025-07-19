import sys

from timecopilot.agent import MODELS
from timecopilot.models.foundational.chronos import Chronos
from timecopilot.models.foundational.moirai import Moirai
from timecopilot.models.foundational.toto import Toto

# Fast benchmark models for regular CI
benchmark_models = [
    "AutoARIMA",
    "SeasonalNaive",
    "ZeroModel",
    "ADIDA",
    "Prophet",
]

# Fast models for regular CI (exclude TimesFM as it's heavy)
fast_models = [MODELS[str_model] for str_model in benchmark_models]

# Add lightweight foundation models for fast tests
if sys.version_info < (3, 13):
    from tabpfn_time_series import TabPFNMode
    from timecopilot.models.foundational.tabpfn import TabPFN
    fast_models.append(TabPFN(mode=TabPFNMode.MOCK))

# Comprehensive model list including heavy foundation models (for slow tests)
all_models = fast_models.copy()

# Add TimesFM to comprehensive list
all_models.append(MODELS["TimesFM"])

# Add TiRex for Python >= 3.11
if sys.version_info >= (3, 11):
    from timecopilot.models.foundational.tirex import TiRex
    all_models.append(TiRex())

# Add heavy foundation models to comprehensive list
all_models.extend(
    [
        Chronos(repo_id="amazon/chronos-t5-tiny", alias="Chronos-T5"),
        Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
        Toto(context_length=256, batch_size=2),
        Moirai(
            context_length=256,
            batch_size=2,
            repo_id="Salesforce/moirai-1.1-R-small",
        ),
        Moirai(
            context_length=256,
            batch_size=2,
            repo_id="Salesforce/moirai-moe-1.0-R-small",
            alias="Moirai-MoE",
        ),
    ]
)

# Default to fast models for regular CI
models = fast_models
