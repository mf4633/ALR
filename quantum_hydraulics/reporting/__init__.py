"""
Quantum Hydraulics -- Professional PDF Report Generator.

Generates PE-stampable engineering reports using reportlab.
"""

from quantum_hydraulics.reporting.report_generator import (
    ReportBuilder,
    ReportConfig,
    generate_scour_report,
    generate_alr_report,
    generate_engineering_report,
    generate_sediment_transport_report,
)

__all__ = [
    "ReportBuilder",
    "ReportConfig",
    "generate_scour_report",
    "generate_alr_report",
    "generate_engineering_report",
    "generate_sediment_transport_report",
]
