"""
Professional PDF Report Generator for Quantum Hydraulics.

Uses reportlab to produce PE-stampable engineering documents with:
- Cover page with project metadata
- Numbered sections and subsections
- Data tables with professional formatting
- Embedded matplotlib figures
- Methodology descriptions
- Limitations disclaimers
- PE signature/seal block
- Optional "PRELIMINARY" watermark
"""

import os
from io import BytesIO
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable,
)
from reportlab.lib.utils import ImageReader


# ── Constants ─────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = letter  # 612 x 792 pt
MARGIN = 72  # 1 inch
CONTENT_W = PAGE_W - 2 * MARGIN  # 468 pt

NAVY = colors.HexColor("#003366")
DARK_GRAY = colors.HexColor("#333333")
LIGHT_GRAY = colors.HexColor("#f0f0f0")
MED_GRAY = colors.HexColor("#888888")
RED_WARN = colors.HexColor("#cc0000")
WHITE = colors.white
BLACK = colors.black


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    """Metadata for report cover page and PE stamp."""
    project_name: str = "Hydraulic Analysis"
    project_number: str = ""
    client: str = ""
    site_location: str = ""
    firm_name: str = ""
    firm_address: str = ""
    prepared_by: str = ""
    reviewed_by: str = ""
    pe_name: str = ""
    pe_license_number: str = ""
    pe_state: str = ""
    report_date: str = ""
    draft: bool = True
    output_path: str = "quantum_report.pdf"


# ── Styles ────────────────────────────────────────────────────────────────

def _build_styles():
    """Create paragraph styles for engineering reports."""
    s = {}
    s["Title"] = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=6,
    )
    s["Subtitle"] = ParagraphStyle(
        "Subtitle", fontName="Helvetica", fontSize=12,
        textColor=DARK_GRAY, alignment=TA_CENTER, spaceAfter=24,
    )
    s["H1"] = ParagraphStyle(
        "H1", fontName="Helvetica-Bold", fontSize=14,
        textColor=NAVY, spaceBefore=18, spaceAfter=8,
        keepWithNext=True,
    )
    s["H2"] = ParagraphStyle(
        "H2", fontName="Helvetica-Bold", fontSize=11,
        textColor=DARK_GRAY, spaceBefore=12, spaceAfter=6,
        keepWithNext=True,
    )
    s["Body"] = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=10,
        textColor=BLACK, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    s["BodySmall"] = ParagraphStyle(
        "BodySmall", fontName="Helvetica", fontSize=9,
        textColor=DARK_GRAY, leading=12, alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    s["Caption"] = ParagraphStyle(
        "Caption", fontName="Helvetica-Oblique", fontSize=9,
        textColor=DARK_GRAY, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=12,
    )
    s["Disclaimer"] = ParagraphStyle(
        "Disclaimer", fontName="Helvetica-Oblique", fontSize=9,
        textColor=RED_WARN, leading=12, alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    s["CoverInfo"] = ParagraphStyle(
        "CoverInfo", fontName="Helvetica", fontSize=10,
        textColor=DARK_GRAY, alignment=TA_LEFT, spaceAfter=2,
    )
    s["PEText"] = ParagraphStyle(
        "PEText", fontName="Helvetica", fontSize=9,
        textColor=BLACK, leading=12, spaceAfter=4,
    )
    s["Reference"] = ParagraphStyle(
        "Reference", fontName="Helvetica", fontSize=9,
        textColor=DARK_GRAY, leading=12, leftIndent=24,
        firstLineIndent=-24, spaceAfter=4,
    )
    return s


# ── Page callbacks ────────────────────────────────────────────────────────

def _make_page_callbacks(config):
    """Return (onFirstPage, onLaterPages) callbacks."""

    def _first_page(canvas, doc):
        canvas.saveState()
        if config.draft:
            _draw_watermark(canvas)
        canvas.restoreState()

    def _later_pages(canvas, doc):
        canvas.saveState()
        # Header rule
        canvas.setStrokeColor(MED_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 16, PAGE_W - MARGIN, PAGE_H - MARGIN + 16)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MED_GRAY)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 20, "QUANTUM HYDRAULICS")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 20,
                               f"Page {doc.page}")
        # Footer
        canvas.drawCentredString(PAGE_W / 2, 36, f"Page {doc.page}")
        if config.draft:
            _draw_watermark(canvas)
        canvas.restoreState()

    return _first_page, _later_pages


def _draw_watermark(canvas):
    """Draw diagonal PRELIMINARY watermark."""
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 50)
    canvas.setFillColor(colors.Color(0.8, 0, 0, alpha=0.06))
    canvas.translate(PAGE_W / 2, PAGE_H / 2)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "PRELIMINARY")
    canvas.drawCentredString(0, -60, "NOT FOR CONSTRUCTION")
    canvas.restoreState()


# ── ReportBuilder ─────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Accumulate report sections as flowables, then build a PDF.

    Usage::

        rb = ReportBuilder(config)
        rb.add_cover_page("SCOUR ASSESSMENT")
        rb.add_section("METHODOLOGY")
        rb.add_paragraph("...")
        rb.add_table(["Col1", "Col2"], [["a", "b"]], caption="Data")
        rb.add_pe_signature_block()
        rb.build()
    """

    def __init__(self, config: ReportConfig):
        self.config = config
        self._elements: list = []
        self._styles = _build_styles()
        self._section_num = 0
        self._subsection_num = 0
        self._figure_num = 0
        self._table_num = 0

    # ── Cover page ────────────────────────────────────────────────────

    def add_cover_page(self, title: str):
        """Add a professional cover page."""
        el = self._elements
        s = self._styles

        el.append(Spacer(1, 80))

        # Firm name
        if self.config.firm_name:
            el.append(Paragraph(self.config.firm_name.upper(),
                                ParagraphStyle("FirmName", fontName="Helvetica-Bold",
                                               fontSize=16, textColor=NAVY,
                                               alignment=TA_CENTER, spaceAfter=4)))
        if self.config.firm_address:
            el.append(Paragraph(self.config.firm_address, s["CoverInfo"]))
            el.append(Spacer(1, 12))

        # Rule
        el.append(HRFlowable(width="80%", color=NAVY, thickness=2,
                              spaceAfter=20, spaceBefore=12))

        # Title
        el.append(Paragraph(title.upper(), s["Title"]))
        el.append(Spacer(1, 12))

        # Project info table
        rdate = self.config.report_date or date.today().strftime("%B %d, %Y")
        info_rows = []
        if self.config.project_name:
            info_rows.append(["Project:", self.config.project_name])
        if self.config.project_number:
            info_rows.append(["Project No.:", self.config.project_number])
        if self.config.client:
            info_rows.append(["Client:", self.config.client])
        if self.config.site_location:
            info_rows.append(["Location:", self.config.site_location])
        info_rows.append(["Date:", rdate])

        if info_rows:
            t = Table(info_rows, colWidths=[1.4 * inch, 4.0 * inch])
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            el.append(t)

        el.append(Spacer(1, 30))

        # Prepared by / Reviewed by
        if self.config.prepared_by:
            el.append(Paragraph(f"Prepared By: {self.config.prepared_by}", s["CoverInfo"]))
        if self.config.reviewed_by:
            el.append(Paragraph(f"Reviewed By: {self.config.reviewed_by}", s["CoverInfo"]))

        el.append(Spacer(1, 40))
        el.append(HRFlowable(width="80%", color=NAVY, thickness=1, spaceAfter=8))
        el.append(Paragraph(
            "Generated by Quantum Hydraulics — Physics-Based Vortex Particle Simulation",
            ParagraphStyle("GenBy", fontName="Helvetica-Oblique", fontSize=8,
                           textColor=MED_GRAY, alignment=TA_CENTER),
        ))
        el.append(PageBreak())

    # ── Sections ──────────────────────────────────────────────────────

    def add_section(self, title: str) -> int:
        self._section_num += 1
        self._subsection_num = 0
        label = f"{self._section_num}. {title.upper()}"
        self._elements.append(Paragraph(label, self._styles["H1"]))
        return self._section_num

    def add_subsection(self, title: str):
        self._subsection_num += 1
        label = f"{self._section_num}.{self._subsection_num} {title}"
        self._elements.append(Paragraph(label, self._styles["H2"]))

    # ── Content ───────────────────────────────────────────────────────

    def add_paragraph(self, text: str, style: str = "Body"):
        self._elements.append(Paragraph(text, self._styles[style]))

    def add_spacer(self, height: float = 12):
        self._elements.append(Spacer(1, height))

    def add_page_break(self):
        self._elements.append(PageBreak())

    # ── Tables ────────────────────────────────────────────────────────

    def add_table(self, headers: List[str], rows: List[List[str]],
                  caption: str = "", col_widths: Optional[List[float]] = None):
        """Add a formatted data table with navy header and alternating rows."""
        self._table_num += 1
        s = self._styles

        data = [headers] + rows
        if col_widths is None:
            n = len(headers)
            col_widths = [CONTENT_W / n] * n

        t = Table(data, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            # Body
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            # Grid
            ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        # Alternating row shading
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))

        t.setStyle(TableStyle(style_cmds))

        block = [t]
        if caption:
            block.append(Paragraph(f"Table {self._table_num}: {caption}", s["Caption"]))

        self._elements.append(KeepTogether(block))
        self._elements.append(Spacer(1, 8))

    # ── Figures ───────────────────────────────────────────────────────

    def add_figure(self, source, caption: str = "",
                   width: Optional[float] = None):
        """Add an embedded figure from a file path or BytesIO."""
        self._figure_num += 1
        s = self._styles

        if width is None:
            width = CONTENT_W * 0.88

        # Get aspect ratio
        try:
            reader = ImageReader(source)
            iw, ih = reader.getSize()
            height = width * (ih / iw)
        except Exception:
            height = width * 0.6  # fallback

        if isinstance(source, str) and not os.path.exists(source):
            self._elements.append(Paragraph(
                f"[Figure {self._figure_num}: {caption} — file not found: {source}]",
                s["Disclaimer"],
            ))
            return

        img = Image(source, width=width, height=height)
        block = [img]
        if caption:
            block.append(Paragraph(f"Figure {self._figure_num}: {caption}", s["Caption"]))

        self._elements.append(KeepTogether(block))
        self._elements.append(Spacer(1, 8))

    # ── Methodology ───────────────────────────────────────────────────

    def add_methodology(self, methodology_type: str = "scour"):
        """Add canned methodology section."""
        sec = self.add_section("Methodology")
        s = self._styles

        if methodology_type == "scour":
            self.add_subsection("Hydraulic Analysis")
            self.add_paragraph(
                "Hydraulic parameters were computed using first-principles fluid mechanics. "
                "The Darcy-Weisbach friction factor was determined iteratively via the "
                "<b>Colebrook-White equation</b>:"
            )
            self.add_paragraph(
                "1/sqrt(f) = -2 log10( ks/(14.8R) + 2.51/(Re sqrt(f)) )",
                "BodySmall",
            )
            self.add_paragraph(
                "where <i>f</i> is the Darcy-Weisbach friction factor, <i>ks</i> is the equivalent "
                "sand roughness, <i>R</i> is the hydraulic radius, and <i>Re</i> is the Reynolds number. "
                "This replaces the empirical Manning's equation with a physics-based approach that "
                "explicitly accounts for boundary layer roughness effects."
            )
            self.add_subsection("Turbulence Analysis")
            self.add_paragraph(
                "Turbulent velocity fields were resolved using the <b>Vortex Particle Method</b> "
                "with Biot-Savart velocity induction. Vortex particles carry three-dimensional "
                "vorticity vectors and interact via the regularized Biot-Savart kernel. Viscous "
                "diffusion is modeled through Particle Strength Exchange (PSE). Velocity profiles "
                "follow the logarithmic law of the wall in the inner layer and a 1/7th power law "
                "in the outer layer."
            )
            self.add_subsection("Scour Risk Assessment")
            self.add_paragraph(
                "Scour risk is evaluated using a non-saturating logistic function of the "
                "excess shear ratio: <b>Risk = 1 / (1 + exp(-2.5(tau/tau_c - 1)))</b>, where "
                "<i>tau</i> is the computed bed shear stress and <i>tau_c</i> is the critical "
                "shear stress for the bed material. Sediment transport capacity is estimated "
                "using the <b>Meyer-Peter Muller</b> formula, and the Shields parameter is computed "
                "as theta = tau / ((rho_s - rho) g d50)."
            )

        elif methodology_type == "alr":
            self.add_subsection("Adaptive Lagrangian Refinement")
            self.add_paragraph(
                "Adaptive Lagrangian Refinement (ALR) employs observation-dependent resolution "
                "to concentrate computational effort where engineering measurements are needed. "
                "Each vortex particle carries an adaptive core size sigma that controls the "
                "spatial resolution of the velocity field reconstruction."
            )
            self.add_subsection("Observation-Dependent Resolution")
            self.add_paragraph(
                "The core size is computed as: <b>sigma = sigma_base / enhancement</b>, where "
                "the enhancement factor is a Gaussian function of distance from the observation zone:"
            )
            self.add_paragraph(
                "enhancement = 1 + 4 exp( -(dist / obs_radius)^2 )",
                "BodySmall",
            )
            self.add_paragraph(
                "This produces up to 5x resolution enhancement at the observation center, with "
                "smooth decay to base resolution at distance. Multiple observation zones are "
                "supported; the maximum enhancement across all zones determines the local sigma."
            )
            self.add_subsection("Physics Engine")
            self.add_paragraph(
                "The underlying hydraulics use Colebrook-White friction (not Manning's equation), "
                "Kolmogorov cascade energy scaling, and full Biot-Savart velocity induction with "
                "Numba-accelerated kernels. The method resolves turbulent vorticity fields from "
                "first principles rather than empirical turbulence models."
            )

        elif methodology_type == "engineering":
            self.add_subsection("Hydraulic Analysis")
            self.add_paragraph(
                "Hydraulic parameters were computed using Colebrook-White friction and "
                "first-principles fluid mechanics, consistent with the scour methodology "
                "described above."
            )
            self.add_subsection("Bank Shear Analysis")
            self.add_paragraph(
                "Bank shear stress was estimated using the USACE bank shear ratio method "
                "(EM 1110-2-1601): <b>\u03c4_bank = K_bank \u00d7 \u03c4_bed</b>, where "
                "K_bank = 0.75 for straight channels. Bank stability is assessed by "
                "comparing computed bank shear to permissible shear for the bank lining "
                "material (bare soil, grass, riprap, etc.)."
            )
            self.add_subsection("Bed Degradation")
            self.add_paragraph(
                "Bed degradation potential was assessed by comparing sediment transport "
                "capacity (Meyer-Peter M\u00fcller formula) between adjacent reaches. A "
                "positive transport deficit (downstream capacity exceeds upstream supply) "
                "indicates degradation. Annual degradation depth is estimated from the "
                "deficit rate over a representative storm duration."
            )
            self.add_subsection("Culvert Outlet Scour")
            self.add_paragraph(
                "Culvert outlet scour was evaluated using a synthetic jet expansion model "
                "with Gaussian lateral velocity spread. Riprap sizing follows Lane's formula: "
                "<b>D\u2085\u2080 = 0.020 V\u00b2</b> (inches) for V < 10 fps, and "
                "D\u2085\u2080 = 0.025 V\u00b2 for V \u2265 10 fps. Apron length is based on "
                "USBR/HEC-14 guidance."
            )
            self.add_subsection("Channel Bend Analysis")
            self.add_paragraph(
                "Bend shear amplification was computed using the forced-vortex velocity "
                "distribution: <b>V(r) = V_mean \u00d7 r/R</b>, which produces higher "
                "velocity and shear stress at the outer bank. Bend scour depth was estimated "
                "using a Lacey-type relationship."
            )

        elif methodology_type == "sediment":
            self.add_subsection("Quasi-Unsteady Approach")
            self.add_paragraph(
                "The hydrograph was discretized into flow-duration records. For each "
                "flow step, normal-depth hydraulics were computed using the Colebrook-White "
                "equation (not Manning's), and sediment transport was evaluated for each "
                "grain-size fraction independently."
            )
            self.add_subsection("Fractional Transport (Meyer-Peter M\u00fcller)")
            self.add_paragraph(
                "Bedload transport was computed per grain-size fraction using the "
                "Meyer-Peter M\u00fcller formula with <b>Egiazaroff hiding/exposure correction</b>. "
                "The hiding exponent shelters fine particles among coarse ones and exposes "
                "coarse particles protruding above the mean bed, capturing the physics of "
                "selective transport in graded beds."
            )
            self.add_subsection("Active Layer Model (Hirano 1971)")
            self.add_paragraph(
                "A thin surface active layer (thickness = 2 \u00d7 d90) exchanges material "
                "with the bulk substrate below. As fine fractions are transported, the "
                "surface coarsens, forming an <b>armor layer</b> that reduces transport "
                "capacity. This self-limiting feedback is the primary control on long-term "
                "bed degradation below dams."
            )
            self.add_subsection("Morphodynamic Feedback (Exner Equation)")
            self.add_paragraph(
                "Bed elevation was updated at each time step via the Exner sediment continuity "
                "equation: <b>dz/dt = (q_s,in - q_s,out) / ((1-p) W)</b>, where p is bed "
                "porosity and W is channel width. Updated bed elevation feeds back to the "
                "hydraulic computation (depth, velocity, shear), creating the morphodynamic "
                "loop that governs long-term channel evolution."
            )

    # ── Limitations ───────────────────────────────────────────────────

    def add_limitations(self):
        """Add standardized limitations and disclaimers section."""
        self.add_section("Limitations and Disclaimers")

        # Red-bordered callout
        warning_text = (
            "<b>IMPORTANT:</b> This analysis was performed using Quantum Hydraulics, "
            "a research-grade computational tool. Results are suitable for screening-level "
            "assessment and preliminary engineering evaluation only. This software has not "
            "been peer-reviewed or approved by FEMA, USACE, or any regulatory agency. "
            "Results should not be used as the sole basis for design without independent "
            "verification using agency-accepted methods (HEC-RAS, MIKE, etc.)."
        )
        warning_para = Paragraph(warning_text, self._styles["Disclaimer"])
        warning_table = Table([[warning_para]], colWidths=[CONTENT_W - 16])
        warning_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1.5, RED_WARN),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        self._elements.append(warning_table)
        self._elements.append(Spacer(1, 12))

        self.add_paragraph(
            "The validation suite includes verification against analytical solutions "
            "(Lamb-Oseen vortex decay, Poiseuille flow, Kolmogorov energy spectrum) and "
            "conservation law checks (continuity, energy). It does not include comparison "
            "to laboratory flume experiments or field measurements at prototype scale."
        )
        self.add_paragraph(
            "<b>Appropriate uses:</b> Screening-level scour risk assessment, relative "
            "comparison of design alternatives, identification of critical locations, "
            "preliminary engineering analysis, and research applications."
        )
        self.add_paragraph(
            "<b>Not appropriate for:</b> FEMA flood studies, NFIP compliance, regulatory "
            "permitting, or final design certification without peer review and independent "
            "verification."
        )

    # ── PE signature block ────────────────────────────────────────────

    def add_pe_signature_block(self):
        """Add Professional Engineer certification and seal placeholder."""
        self._elements.append(Spacer(1, 24))
        self._elements.append(HRFlowable(width="100%", color=NAVY, thickness=1.5,
                                          spaceAfter=12))
        self._elements.append(Paragraph(
            "PROFESSIONAL ENGINEER CERTIFICATION",
            self._styles["H1"],
        ))

        pe_state = self.config.pe_state or "________"
        pe_name = self.config.pe_name or "________________________________"
        pe_lic = self.config.pe_license_number or "________________"

        cert_text = (
            f"I hereby certify that this engineering document was prepared by me or "
            f"under my direct supervision and that I am a duly licensed Professional "
            f"Engineer under the laws of the State of {pe_state}."
        )

        # Left column: text fields
        left_content = [
            [Paragraph(cert_text, self._styles["PEText"])],
            [Spacer(1, 16)],
            [Paragraph("Signature: ________________________________", self._styles["PEText"])],
            [Spacer(1, 8)],
            [Paragraph("Date: ________________________________", self._styles["PEText"])],
            [Spacer(1, 8)],
            [Paragraph(f"Name: {pe_name}", self._styles["PEText"])],
            [Paragraph(f"P.E. License No. {pe_lic}", self._styles["PEText"])],
        ]
        left_table = Table(left_content, colWidths=[3.2 * inch])
        left_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))

        # Right column: seal placeholder
        seal_text = Paragraph(
            "<br/><br/>PLACE PE SEAL HERE<br/><br/>"
            "<font size='7'>This space reserved for<br/>"
            "Professional Engineer's seal</font>",
            ParagraphStyle("SealText", fontName="Helvetica", fontSize=9,
                           textColor=MED_GRAY, alignment=TA_CENTER),
        )
        seal_box = Table([[seal_text]], colWidths=[2.0 * inch], rowHeights=[2.0 * inch])
        seal_box.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, MED_GRAY),
            ("LINESTYLE", (0, 0), (-1, -1), "DASHED"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        # Combine left + right
        layout = Table([[left_table, seal_box]],
                       colWidths=[3.5 * inch, 2.5 * inch])
        layout.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        self._elements.append(layout)

    # ── Build ─────────────────────────────────────────────────────────

    def build(self) -> str:
        """Render PDF and return the output file path."""
        doc = SimpleDocTemplate(
            self.config.output_path,
            pagesize=letter,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
        )
        first_page, later_pages = _make_page_callbacks(self.config)
        doc.build(self._elements, onFirstPage=first_page, onLaterPages=later_pages)
        return self.config.output_path


# ── Convenience: Scour Report ─────────────────────────────────────────────

def generate_scour_report(
    design_results=None,
    quantum_node=None,
    figures: Optional[Dict[str, str]] = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """
    Generate a screening-level scour assessment PDF.

    Parameters
    ----------
    design_results : DesignResults, optional
        From quantum_hydraulics.analysis.analyze()
    quantum_node : QuantumNode, optional
        After calling compute_turbulence()
    figures : dict, optional
        {"label": "path.png"} for embedding
    config : ReportConfig, optional

    Returns
    -------
    str
        Path to generated PDF
    """
    if config is None:
        config = ReportConfig(output_path="Scour_Assessment_Report.pdf")

    rb = ReportBuilder(config)
    rb.add_cover_page("Screening-Level Scour Assessment")

    # 1. Introduction
    rb.add_section("Introduction")
    rb.add_subsection("Purpose")
    rb.add_paragraph(
        "This report presents the results of a screening-level scour assessment "
        "performed using physics-based vortex particle simulation. The analysis "
        "evaluates bed shear stress, scour risk, sediment transport potential, and "
        "provides preliminary recommendations for scour countermeasures."
    )
    if config.site_location:
        rb.add_subsection("Site Description")
        rb.add_paragraph(f"The analysis was performed for: {config.site_location}.")

    # 2. Methodology
    rb.add_methodology("scour")

    # 3. Input Parameters
    if design_results:
        dr = design_results
        rb.add_section("Input Parameters")
        rb.add_table(
            ["Parameter", "Value", "Units"],
            [
                ["Discharge (Q)", f"{dr.Q:.1f}", "cfs"],
                ["Channel Width", f"{dr.width:.1f}", "ft"],
                ["Water Depth", f"{dr.depth:.2f}", "ft"],
                ["Bed Slope", f"{dr.slope:.4f}", "ft/ft"],
                ["Roughness (ks)", f"{dr.roughness_ks:.3f}", "ft"],
            ],
            caption="Design Input Parameters",
            col_widths=[2.5 * inch, 2.0 * inch, 1.0 * inch],
        )

    # 4. Results
    rb.add_section("Results")

    if design_results:
        dr = design_results
        rb.add_subsection("Hydraulic Results")
        rb.add_table(
            ["Parameter", "Value", "Units"],
            [
                ["Mean Velocity", f"{dr.velocity_mean:.2f}", "ft/s"],
                ["Maximum Velocity", f"{dr.velocity_max:.2f}", "ft/s"],
                ["Cross-Section Area", f"{dr.area:.1f}", "ft\u00b2"],
                ["Hydraulic Radius", f"{dr.hydraulic_radius:.2f}", "ft"],
                ["Reynolds Number", f"{dr.reynolds_number:,.0f}", "\u2014"],
                ["Froude Number", f"{dr.froude_number:.3f}", "\u2014"],
                ["Friction Factor (f)", f"{dr.friction_factor:.5f}", "\u2014"],
                ["Flow Regime", dr.flow_regime, "\u2014"],
            ],
            caption="Computed Hydraulic Parameters",
            col_widths=[2.5 * inch, 2.0 * inch, 1.0 * inch],
        )

        rb.add_subsection("Turbulence and Shear")
        rb.add_table(
            ["Parameter", "Value", "Units"],
            [
                ["Friction Velocity (u*)", f"{dr.friction_velocity:.4f}", "ft/s"],
                ["Bed Shear Stress", f"{dr.bed_shear_stress:.4f}", "psf"],
                ["Turbulent Kinetic Energy", f"{dr.tke:.4f}", "ft\u00b2/s\u00b2"],
                ["Kolmogorov Scale", f"{dr.kolmogorov_scale:.6f}", "ft"],
            ],
            caption="Turbulence and Shear Parameters",
            col_widths=[2.5 * inch, 2.0 * inch, 1.0 * inch],
        )

        rb.add_subsection("Scour Assessment")
        rb.add_table(
            ["Parameter", "Value"],
            [
                ["Scour Risk Index (0-1)", f"{dr.scour_risk_index:.3f}"],
                ["Scour Assessment", dr.scour_assessment],
                ["Velocity Assessment", dr.velocity_assessment],
            ],
            caption="Design Assessments",
            col_widths=[2.5 * inch, 4.0 * inch],
        )

    if quantum_node and hasattr(quantum_node, 'metrics') and quantum_node.metrics:
        m = quantum_node.metrics
        rb.add_subsection("Vortex Particle Analysis")
        rb.add_table(
            ["Parameter", "Value", "Units"],
            [
                ["Max Velocity", f"{m.max_velocity:.2f}", "ft/s"],
                ["Mean Velocity", f"{m.mean_velocity:.2f}", "ft/s"],
                ["Bed Shear Stress", f"{m.bed_shear_stress:.4f}", "psf"],
                ["Scour Risk Index", f"{m.scour_risk_index:.3f}", "\u2014"],
                ["Shields Parameter", f"{m.shields_parameter:.4f}", "\u2014"],
                ["Excess Shear Ratio", f"{m.excess_shear_ratio:.2f}", "\u2014"],
                ["Sediment Transport", f"{m.sediment_transport_rate:.6f}", "lb/ft/s"],
                ["Scour Depth Potential", f"{m.scour_depth_potential:.2f}", "ft/yr"],
            ],
            caption="Vortex Particle Turbulence Results",
            col_widths=[2.5 * inch, 2.0 * inch, 1.0 * inch],
        )

        # Text assessments
        rb.add_paragraph(f"<b>Scour Assessment:</b> {quantum_node.get_scour_assessment()}")
        rb.add_paragraph(f"<b>Velocity Assessment:</b> {quantum_node.get_velocity_assessment()}")
        rb.add_paragraph(
            f"<b>Sediment Transport:</b> {quantum_node.get_sediment_transport_assessment()}"
        )

        # Energy dissipation recommendations
        rec = quantum_node.get_energy_dissipation_recommendation()
        if rec.get("recommended_dissipator") != "None required":
            rb.add_subsection("Energy Dissipation Recommendations")
            rb.add_table(
                ["Parameter", "Value"],
                [
                    ["Recommended Dissipator", rec.get("recommended_dissipator", "N/A")],
                    ["Riprap D50", f"{rec.get('riprap_d50_inches', 0):.1f} inches"],
                    ["Apron Length", f"{rec.get('apron_length_ft', 0):.1f} ft"],
                    ["Energy to Dissipate", f"{rec.get('energy_to_dissipate_ft', 0):.2f} ft"],
                ],
                caption="Energy Dissipation Design Recommendations",
                col_widths=[2.5 * inch, 4.0 * inch],
            )

    # 5. Figures
    if figures:
        rb.add_section("Figures")
        for label, path in figures.items():
            rb.add_figure(path, caption=label)

    # 6. Conclusions
    rb.add_section("Conclusions and Recommendations")
    if design_results:
        risk = design_results.scour_risk_index
        if risk > 0.8:
            rb.add_paragraph(
                "The analysis indicates <b>critical scour risk</b>. Scour protection "
                "measures are required. It is recommended that a detailed scour analysis "
                "be performed using agency-accepted methods (HEC-RAS, HEC-18) to confirm "
                "these findings and design appropriate countermeasures."
            )
        elif risk > 0.5:
            rb.add_paragraph(
                "The analysis indicates <b>elevated scour risk</b>. Scour protection "
                "is recommended. Further analysis using agency-accepted methods is advised "
                "to confirm findings and develop detailed countermeasure designs."
            )
        elif risk > 0.2:
            rb.add_paragraph(
                "The analysis indicates <b>moderate scour risk</b>. Monitoring is recommended. "
                "Consider further analysis if conditions change or if the structure is critical."
            )
        else:
            rb.add_paragraph(
                "The analysis indicates <b>low scour risk</b> under the evaluated conditions. "
                "Routine monitoring is recommended."
            )

    # 7. Limitations
    rb.add_limitations()

    # PE block
    if config.pe_name:
        rb.add_pe_signature_block()

    return rb.build()


# ── Convenience: ALR Report ───────────────────────────────────────────────

def generate_alr_report(
    convergence=None,
    cost_benefit=None,
    sigma_field=None,
    scour=None,
    multi_zone=None,
    figure_dir: str = "ALR_figures",
    config: Optional[ReportConfig] = None,
) -> str:
    """
    Generate ALR research report PDF.

    Parameters
    ----------
    convergence : ConvergenceResult, optional
    cost_benefit : CostBenefitResult, optional
    sigma_field : SigmaFieldResult, optional
    scour : ScourResult, optional
    multi_zone : MultiZoneResult, optional
    figure_dir : str
        Directory containing ALR figure PNGs
    config : ReportConfig, optional

    Returns
    -------
    str
        Path to generated PDF
    """
    if config is None:
        config = ReportConfig(
            project_name="ALR Research Study",
            output_path="ALR_Research_Report.pdf",
            draft=False,
        )

    rb = ReportBuilder(config)
    rb.add_cover_page("Adaptive Lagrangian Refinement\nTechnical Summary")

    # 1. Introduction
    rb.add_section("Introduction")
    rb.add_paragraph(
        "This report presents the results of Adaptive Lagrangian Refinement (ALR) "
        "experiments conducted using the Quantum Hydraulics vortex particle simulation "
        "package. ALR employs observation-dependent resolution to concentrate "
        "computational effort at engineering-critical locations while maintaining "
        "coarse approximation elsewhere."
    )
    rb.add_paragraph(
        "The key innovation is that computational cost scales with where engineers "
        "observe, not with domain size. This enables practical application of "
        "physics-based turbulence simulation to engineering screening problems that "
        "would otherwise require prohibitively expensive uniform high-resolution computation."
    )

    # 2. Methodology
    rb.add_methodology("alr")

    # 3. Convergence
    if convergence:
        r = convergence
        rb.add_section("Experiment 1: Convergence Study")
        rb.add_paragraph(
            "This experiment verifies that ALR metrics at the observation zone converge "
            "as the observation radius increases, approaching uniform high-resolution results."
        )
        rows = []
        for i, rad in enumerate(r.obs_radii):
            rows.append([
                f"{rad:.0f}",
                f"{r.mean_sigma[i]:.4f}",
                f"{r.mean_vorticity[i]:.4f}",
                f"{r.mean_enstrophy[i]:.4f}",
                str(r.n_particles[i]),
            ])
        rb.add_table(
            ["Obs Radius (ft)", "Mean Sigma", "Mean Vorticity", "Mean Enstrophy", "Particles in Box"],
            rows,
            caption="Convergence of ALR Metrics vs. Observation Radius",
        )
        fig_path = os.path.join(figure_dir, "fig2_convergence.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Convergence of core size and vorticity with observation radius")

    # 4. Cost-Benefit
    if cost_benefit:
        r = cost_benefit
        rb.add_section("Experiment 2: Cost-Benefit Analysis")
        rb.add_paragraph(
            f"This experiment compares ALR at various particle counts against a uniform "
            f"high-resolution baseline (6,000 particles, all sigmas at minimum). "
            f"Baseline vorticity: {r.baseline_vorticity:.4f}."
        )
        rows = []
        for i, n in enumerate(r.particle_counts):
            rows.append([
                f"{n:,}",
                f"{r.errors_vorticity[i]:.1%}",
                f"{r.errors_enstrophy[i]:.1%}",
                f"{r.wall_times[i]:.2f}",
            ])
        rb.add_table(
            ["Particle Count", "Vorticity Error", "Enstrophy Error", "Wall Time (s)"],
            rows,
            caption="ALR Cost-Benefit: Accuracy vs. Computational Effort",
        )
        fig_path = os.path.join(figure_dir, "fig3_cost_benefit.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Cost-benefit analysis: accuracy and wall time vs. particle count")

    # 5. Sigma Field
    if sigma_field:
        r = sigma_field
        rb.add_section("Experiment 3: Sigma Field Visualization")
        rb.add_paragraph(
            f"This experiment visualizes the adaptive resolution field under three "
            f"configurations. The observation-dependent enhancement at the pier wake center "
            f"is <b>{r.enhancement_at_center:.1f}x</b>, meaning the core size sigma is reduced "
            f"by that factor relative to the domain boundary."
        )
        fig_path = os.path.join(figure_dir, "fig1_sigma_field.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path,
                          "Sigma fields: observation at pier wake (left), "
                          "at channel entrance (center), off (right)")

    # 6. Scour
    if scour:
        r = scour
        rb.add_section("Experiment 4: Engineering Scour Validation")
        rb.add_paragraph(
            "This experiment validates the engineering relevance of the vortex particle "
            "method by comparing Tier 1 (vectorized Colebrook-White) and Tier 2 (full "
            "Biot-Savart vortex particle) analyses at a synthetic bridge pier."
        )
        rb.add_table(
            ["Metric", "Value"],
            [
                ["Tier 1 Approach Shear", f"{r.tier1_shear_approach:.4f} psf"],
                ["Tier 1 Pier Shear", f"{r.tier1_shear_pier:.4f} psf"],
                ["Tier 2 Pier Shear (Quantum)", f"{r.tier2_shear_pier:.4f} psf"],
                ["Amplification Factor", f"{r.amplification:.2f}x"],
                ["Scour Risk Index", f"{r.tier2_scour_risk:.3f}"],
                ["Shields Parameter", f"{r.tier2_shields:.4f}"],
                ["Hotspot Cells Analyzed", f"{r.n_hotspots}"],
            ],
            caption="Tier 1 vs. Tier 2 Scour Analysis at Bridge Pier",
            col_widths=[3.0 * inch, 3.5 * inch],
        )
        fig_path = os.path.join(figure_dir, "fig4_pier_scour.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Bed shear stress comparison: approach, pier (Tier 1), pier (Tier 2)")

    # 7. Multi-Zone
    if multi_zone:
        r = multi_zone
        rb.add_section("Experiment 5: Multi-Zone Independence")
        rb.add_paragraph(
            "This experiment verifies that multiple observation zones operate independently "
            "on a 400-ft channel with zones at x=100 ft and x=300 ft."
        )
        rb.add_table(
            ["Location", "Mean Sigma", "Mean Vorticity"],
            [
                ["Zone A (x=100)", f"{r.zone_a_sigma:.4f}", f"{r.zone_a_vorticity:.4f}"],
                ["Zone B (x=300)", f"{r.zone_b_sigma:.4f}", f"{r.zone_b_vorticity_base:.4f}"],
                ["Midpoint (x=200)", f"{r.midpoint_sigma:.4f}", "\u2014"],
                ["Zone B (shifted to x=350)", "\u2014", f"{r.zone_b_vorticity_moved:.4f}"],
            ],
            caption="Multi-Zone Resolution and Independence Verification",
            col_widths=[2.5 * inch, 2.0 * inch, 2.0 * inch],
        )
        fig_path = os.path.join(figure_dir, "fig5_multi_zone.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Dual observation zones with independent resolution concentration")

    # 8. Conclusions
    rb.add_section("Conclusions")
    conclusions = []
    if convergence:
        conclusions.append(
            "ALR metrics converge as the observation radius increases, confirming that "
            "observation-dependent resolution produces consistent results."
        )
    if cost_benefit:
        conclusions.append(
            "ALR achieves comparable vorticity accuracy to uniform high-resolution simulation "
            "at significantly reduced particle counts, demonstrating substantial computational savings."
        )
    if sigma_field:
        conclusions.append(
            f"The adaptive sigma field concentrates resolution at observation zones with up to "
            f"{sigma_field.enhancement_at_center:.0f}x enhancement, while maintaining uniform "
            f"resolution when observation is disabled."
        )
    if scour:
        conclusions.append(
            f"Vortex particle analysis (Tier 2) amplifies bed shear by {scour.amplification:.1f}x "
            f"relative to vectorized hydraulics (Tier 1), capturing turbulence-induced scour "
            f"effects at the bridge pier."
        )
    if multi_zone:
        conclusions.append(
            "Multiple observation zones operate independently, enabling simultaneous "
            "high-resolution analysis at multiple critical locations."
        )

    for c in conclusions:
        rb.add_paragraph(f"\u2022 {c}")

    # 9. References
    rb.add_section("References")
    refs = [
        "Colebrook, C.F. (1939). \"Turbulent flow in pipes, with particular reference to the "
        "transition region between smooth and rough pipe laws.\" J. Inst. Civil Engineers, 11(4), 133-156.",
        "Kolmogorov, A.N. (1941). \"The local structure of turbulence in incompressible viscous "
        "fluid for very large Reynolds numbers.\" Dokl. Akad. Nauk SSSR, 30, 301-305.",
        "Cottet, G.H. and Koumoutsakos, P. (2000). \"Vortex Methods: Theory and Practice.\" "
        "Cambridge University Press.",
        "Meyer-Peter, E. and Muller, R. (1948). \"Formulas for bed-load transport.\" "
        "Proc. 2nd Meeting, IAHR, Stockholm, 39-64.",
    ]
    for ref in refs:
        rb.add_paragraph(ref, "Reference")

    # Limitations
    rb.add_limitations()

    # PE block (optional)
    if config.pe_name:
        rb.add_pe_signature_block()

    return rb.build()


# ── Convenience: Engineering Scenarios Report ─────────────────────────────

def generate_engineering_report(
    bank_erosion=None,
    degradation=None,
    culvert_outlet=None,
    bend=None,
    figure_dir: str = "Engineering_figures",
    config: Optional[ReportConfig] = None,
) -> str:
    """
    Generate engineering scenarios assessment PDF.

    Parameters
    ----------
    bank_erosion : dict, optional
        Contains "bare", "grass", "low", "meta" from bank erosion checks
    degradation : DegradationAssessment, optional
    culvert_outlet : CulvertOutletAssessment, optional
    bend : BendAssessment, optional
    figure_dir : str
    config : ReportConfig, optional

    Returns
    -------
    str
        Path to generated PDF
    """
    if config is None:
        config = ReportConfig(output_path="Engineering_Scenarios_Report.pdf")

    rb = ReportBuilder(config)
    rb.add_cover_page("Engineering Scenario Assessment")

    # 1. Introduction
    rb.add_section("Introduction")
    rb.add_paragraph(
        "This report presents screening-level assessments for four common hydraulic "
        "engineering scenarios using physics-based vortex particle simulation. Each "
        "scenario evaluates bed shear, bank stability, sediment transport, or scour "
        "risk using first-principles hydraulics (Colebrook-White friction, Meyer-Peter "
        "M\u00fcller transport) rather than empirical Manning's equation."
    )

    # 2. Methodology
    rb.add_methodology("engineering")

    # 3. Bank Erosion
    if bank_erosion:
        ba_bare = bank_erosion["bare"]
        ba_grass = bank_erosion["grass"]
        rb.add_section("Bank Erosion Assessment")
        rb.add_paragraph(
            "A trapezoidal channel (30 ft bottom width, 3:1 side slopes, 4 ft depth) "
            "was analyzed at bankfull conditions. Bank shear was computed using the USACE "
            f"bank-to-bed shear ratio K_bank = {ba_bare.k_bank}."
        )
        rb.add_table(
            ["Bank Material", "Permissible (psf)", "Computed (psf)", "FOS", "Assessment"],
            [
                ["Bare Soil", f"{ba_bare.permissible_shear:.3f}",
                 f"{ba_bare.max_bank_shear:.4f}", f"{ba_bare.factor_of_safety:.3f}",
                 ba_bare.assessment],
                ["Good Grass", f"{ba_grass.permissible_shear:.3f}",
                 f"{ba_grass.max_bank_shear:.4f}", f"{ba_grass.factor_of_safety:.3f}",
                 ba_grass.assessment],
            ],
            caption="Bank Stability Assessment at Bankfull",
        )
        fig_path = os.path.join(figure_dir, "fig1_bank_shear_plan.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Bank shear stress distribution at bankfull")
        fig_path = os.path.join(figure_dir, "fig2_bank_stages.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Bank shear vs. flow stage with permissible shear thresholds")

    # 4. Bed Degradation
    if degradation:
        d = degradation
        rb.add_section("Bed Degradation Assessment")
        rb.add_paragraph(
            "A 500-ft rectangular channel with a slope break (0.001 upstream, 0.003 "
            "downstream) was analyzed. The steeper downstream reach produces higher "
            "transport capacity, creating a sediment deficit."
        )
        rb.add_table(
            ["Parameter", "Upstream", "Downstream"],
            [
                ["Mean Velocity (fps)", f"{d.upstream_mean_v:.2f}", f"{d.downstream_mean_v:.2f}"],
                ["Transport Capacity (lb/ft/s)", f"{d.upstream_transport:.6f}",
                 f"{d.downstream_transport:.6f}"],
            ],
            caption="Reach-Averaged Hydraulic and Transport Parameters",
            col_widths=[2.5 * inch, 2.0 * inch, 2.0 * inch],
        )
        rb.add_paragraph(
            f"<b>Transport Deficit:</b> {d.transport_deficit:.6f} lb/ft/s<br/>"
            f"<b>Annual Degradation:</b> {d.annual_degradation_ft:.3f} ft/yr<br/>"
            f"<b>Assessment:</b> {d.assessment}"
        )
        fig_path = os.path.join(figure_dir, "fig3_degradation_profile.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path,
                          "Longitudinal transport capacity profile showing deficit at grade break")

    # 5. Culvert Outlet
    if culvert_outlet:
        co = culvert_outlet
        rb.add_section("Culvert Outlet Scour Assessment")
        rb.add_paragraph(
            f"A 6-ft wide culvert discharging at {co.jet_exit_velocity:.1f} fps into a "
            f"40-ft receiving channel was analyzed for outlet scour."
        )
        rb.add_table(
            ["Parameter", "Value"],
            [
                ["Jet Exit Velocity", f"{co.jet_exit_velocity:.2f} fps"],
                ["Max Plunge Shear", f"{co.max_plunge_shear:.4f} psf"],
                ["Required Riprap D50", f"{co.required_riprap_d50_in:.1f} inches"],
                ["Required Apron Length", f"{co.required_apron_length_ft:.1f} ft"],
                ["Scour Risk Index", f"{co.scour_risk:.3f}"],
                ["Assessment", co.assessment],
            ],
            caption="Culvert Outlet Scour Assessment",
            col_widths=[3.0 * inch, 3.5 * inch],
        )
        fig_path = os.path.join(figure_dir, "fig4_culvert_plan.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Culvert outlet velocity field with jet expansion")
        fig_path = os.path.join(figure_dir, "fig5_culvert_scour.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path, "Centerline bed shear stress vs. distance from outlet")

    # 6. Channel Bend
    if bend:
        ba = bend
        rb.add_section("Channel Bend Assessment")
        rb.add_paragraph(
            f"A 90-degree channel bend (R = 100 ft, W = 30 ft, R/W = {ba.r_over_w:.1f}) "
            f"was analyzed for outer bank shear amplification."
        )
        rb.add_table(
            ["Parameter", "Value"],
            [
                ["Approach Shear", f"{ba.approach_mean_shear:.4f} psf"],
                ["Outer Bank Shear", f"{ba.outer_mean_shear:.4f} psf"],
                ["Inner Bank Shear", f"{ba.inner_mean_shear:.4f} psf"],
                ["Amplification Factor", f"{ba.amplification_factor:.3f}x"],
                ["Bend Scour Depth", f"{ba.bend_scour_depth_ft:.2f} ft"],
                ["Assessment", ba.assessment],
            ],
            caption="Channel Bend Shear Assessment",
            col_widths=[3.0 * inch, 3.5 * inch],
        )
        fig_path = os.path.join(figure_dir, "fig6_bend_shear.png")
        if os.path.exists(fig_path):
            rb.add_figure(fig_path,
                          "Bed shear stress in channel bend (outer bank amplification)")

    # 7. Summary
    rb.add_section("Summary")
    summary_rows = []
    if bank_erosion:
        summary_rows.append(["Bank Erosion", bank_erosion["bare"].assessment,
                             f"FOS={bank_erosion['bare'].factor_of_safety:.2f} (bare soil)"])
    if degradation:
        summary_rows.append(["Bed Degradation", degradation.assessment,
                             f"{degradation.annual_degradation_ft:.2f} ft/yr"])
    if culvert_outlet:
        summary_rows.append(["Culvert Outlet", culvert_outlet.assessment,
                             f"D50={culvert_outlet.required_riprap_d50_in:.0f} in"])
    if bend:
        summary_rows.append(["Channel Bend", bend.assessment,
                             f"{bend.amplification_factor:.2f}x amplification"])
    if summary_rows:
        rb.add_table(
            ["Scenario", "Assessment", "Key Metric"],
            summary_rows,
            caption="Engineering Scenario Summary",
        )

    # 8. Limitations
    rb.add_limitations()

    if config.pe_name:
        rb.add_pe_signature_block()

    return rb.build()


# ── Convenience: Sediment Transport Report ────────────────────────────────

def generate_sediment_transport_report(
    results=None,
    figure_dir: str = "Sediment_figures",
    config: Optional[ReportConfig] = None,
) -> str:
    """Generate quasi-unsteady sediment transport assessment PDF."""
    if config is None:
        config = ReportConfig(output_path="Sediment_Transport_Report.pdf")

    rb = ReportBuilder(config)
    rb.add_cover_page("Quasi-Unsteady Sediment Transport Assessment")

    rb.add_section("Introduction")
    rb.add_paragraph(
        "This report presents the results of a quasi-unsteady sediment transport "
        "simulation using fractional bedload transport with active-layer armoring. "
        "The analysis steps through a design hydrograph computing bed elevation "
        "changes, surface coarsening, and armor development."
    )

    rb.add_methodology("sediment")

    if results:
        r = results
        rb.add_section("Input Parameters")
        rb.add_table(
            ["Parameter", "Value"],
            [
                ["Channel Length", f"{r.channel_length:.0f} ft"],
                ["Channel Width", f"{r.channel_width:.0f} ft"],
                ["Initial d50", f"{r.initial_gradation.d50_mm:.3f} mm"],
                ["Initial d90", f"{r.initial_gradation.d90_mm:.3f} mm"],
                ["Number of Grain Fractions", f"{r.initial_gradation.n_fractions}"],
                ["Simulation Steps", f"{len(r.steps)}"],
            ],
            caption="Simulation Input Parameters",
            col_widths=[3.0 * inch, 3.5 * inch],
        )

        rb.add_section("Results")
        rb.add_table(
            ["Metric", "Value"],
            [
                ["Total Bed Change", f"{r.total_scour_ft:.4f} ft"],
                ["Maximum Scour", f"{abs(r.max_scour_ft):.4f} ft"],
                ["Final Surface d50", f"{r.final_d50_mm:.3f} mm"],
                ["Coarsening Ratio", f"{r.final_d50_mm / r.initial_gradation.d50_mm:.2f}x"],
                ["Armoring", "Yes" if r.armored else "No"],
                ["Assessment", r.get_assessment()],
            ],
            caption="Sediment Transport Results Summary",
            col_widths=[3.0 * inch, 3.5 * inch],
        )

        # Figures
        for fname, caption in [
            ("fig1_hydrograph_bed.png", "Hydrograph and bed elevation response"),
            ("fig2_d50_evolution.png", "Surface d50 coarsening over time"),
            ("fig3_cumulative_scour.png", "Cumulative bed change with assessment"),
            ("fig4_transport_rate.png", "Bedload transport rate showing armoring feedback"),
        ]:
            fig_path = os.path.join(figure_dir, fname)
            if os.path.exists(fig_path):
                rb.add_figure(fig_path, caption)

    rb.add_section("Conclusions")
    if results:
        rb.add_paragraph(
            f"The simulation produced {abs(results.total_scour_ft):.3f} ft of cumulative "
            f"bed degradation over the design hydrograph. The bed surface coarsened from "
            f"d50 = {results.initial_gradation.d50_mm:.2f} mm to "
            f"{results.final_d50_mm:.2f} mm "
            f"({results.final_d50_mm / results.initial_gradation.d50_mm:.1f}x), "
            f"{'forming a stable armor layer that limited further scour.' if results.armored else 'but armoring was not fully developed.'}"
        )

    rb.add_limitations()

    if config.pe_name:
        rb.add_pe_signature_block()

    return rb.build()
