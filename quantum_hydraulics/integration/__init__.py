"""
Integration modules for PCSWMM and other hydraulic software.

Contains:
- QuantumNode: Unified node analysis for SWMM junctions
- pcswmm_script: Auto-detect PCSWMM integration script
"""

try:
    from quantum_hydraulics.integration.swmm_node import QuantumNode
    _HAS_SWMM = True
except ImportError:
    _HAS_SWMM = False
    QuantumNode = None

__all__ = ["QuantumNode", "_HAS_SWMM"]
