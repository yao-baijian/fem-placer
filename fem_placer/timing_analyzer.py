
"""
Compatibility re-export module.

``timing_analyzer`` has been split into three modules:

- ``timer`` — ``TimingSummary``, ``TimingAnalyzer``, convenience functions
- ``vivado_timer`` — ``VivadoTimingRunner``, ``generate_vivado_timing_tcl``
- ``rapidwright_timer`` — ``RapidWrightTimer``

All public names are re-exported here so existing imports still work.
"""
import warnings
warnings.warn(
    "timing_analyzer is deprecated — import from timer or vivado_timer instead",
    DeprecationWarning, stacklevel=2,
)

from .timer import (
    TimingSummary,
    TimingAnalyzer,
    analyze_placement_timing,
    analyze_path_based_timing,
    parse_vivado_timing,
)
from .vivado_timer import (
    VivadoTimingRunner,
    generate_vivado_timing_tcl,
)
from .rapidwright_timer import (
    RapidWrightTimer,
)
