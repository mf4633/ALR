"""
Integration modules for PCSWMM and other hydraulic software.

Contains:
- QuantumNode: Unified node analysis for SWMM junctions (1D)
- SWMM2DPostProcessor: 2D mesh post-processor for PCSWMM exports
- pcswmm_script: Auto-detect PCSWMM integration script
"""

try:
    from quantum_hydraulics.integration.swmm_node import QuantumNode
    _HAS_SWMM = True
except ImportError:
    _HAS_SWMM = False
    QuantumNode = None

try:
    from quantum_hydraulics.integration.swmm_2d import SWMM2DPostProcessor
    _HAS_SWMM_2D = True
except ImportError:
    _HAS_SWMM_2D = False
    SWMM2DPostProcessor = None

__all__ = ["QuantumNode", "SWMM2DPostProcessor", "_HAS_SWMM", "_HAS_SWMM_2D"]
