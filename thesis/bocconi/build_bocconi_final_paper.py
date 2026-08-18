"""Build the Bocconi-compliant Bachelor Final Paper DOCX.

The source of truth remains ``thesis/MASTER_DRAFT.md``.  This builder removes
the title page, abstract, and executive summary from the submitted body,
because Bocconi generates the title page and collects the abstract separately.
It then uses Pandoc for citations/equations and applies the applicable 2025-2026
Bachelor Final Paper layout contract in OOXML.
"""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "thesis"
HERE = THESIS / "bocconi"
SOURCE = THESIS / "MASTER_DRAFT.md"
BIBLIOGRAPHY = THESIS / "references.bib"
BODY_MD = HERE / "BOCCONI_FINAL_PAPER.md"
PANDOC_DOCX = HERE / "_pandoc_body.docx"
OUTPUT_DOCX = Path(
    os.environ.get(
        "BOCCONI_OUTPUT_DOCX",
        HERE / "Bocconi_Bachelor_Final_Paper.docx",
    )
)
PANDOC = Path("/opt/anaconda3/bin/pandoc")

# Bocconi 2025-2026 Bachelor Final Paper tokens.
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
MARGIN_SIDE_CM = 2.5
MARGIN_TOP_CM = 2.5
MARGIN_BOTTOM_CM = 2.5
USABLE_WIDTH_DXA = 9072  # 16.0 cm
TABLE_INDENT_DXA = 90
BODY_FONT = "Arial"
BODY_SIZE_PT = 12
BODY_LINE_PT = 24  # about 29 lines on an A4 text page
INK = RGBColor(0, 0, 0)
LIGHT_GRAY = "E7E7E7"

# Static chapter-level contents.  Bocconi requires the body to begin after the
# four blank pages; LibreOffice does not populate an empty Word TOC field in a
# headless conversion, so the submission copy uses a stable, explicit list.
# Values are verified against the rendered PDF in the build QA.
CONTENTS_ENTRIES = [
    ("1. Introduction", 2),
    ("2. Related literature and empirical motivation", 6),
    ("3. Data and sample construction", 11),
    ("4. Predictive methodology", 16),
    ("5. Predictive results", 18),
    ("6. Portfolio-formation methodology", 19),
    ("7. Economic results", 23),
    ("8. Robustness and alternative approaches", 27),
    ("9. Limitations", 30),
    ("10. Conclusion", 31),
    ("References", 33),
]


def _style_by_name(styles, name: str):
    """Return a style by its visible name.

    Pandoc's reference document uses style IDs such as ``Heading1`` while
    exposing visible names such as ``Heading 1``.  Some python-docx releases
    only resolve the ID even though the public API documents name lookup.
    Iterating keeps this builder portable across both behaviours.
    """
    for style in styles:
        if style.name == name:
            return style
    raise KeyError(f"no style with name {name!r}")


def _has_style(styles, name: str) -> bool:
    return any(style.name == name for style in styles)


def _extract_submission_body() -> str:
    text = SOURCE.read_text()
    start = text.index("## 1. Introduction")
    end = text.index("## Appendix A.", start)
    body = text[start:end]
    # Retain the compact, source-backed figure set. Each visual answers a
    # distinct empirical question and complements rather than replaces the
    # exact result tables.
    # Promote manuscript chapter levels for the standalone final paper.
    lines = []
    for line in body.splitlines():
        if line.startswith("### "):
            line = "## " + line[4:]
        elif line.startswith("## "):
            line = "# " + line[3:]
        lines.append(line)
    body = "\n".join(lines).strip() + "\n"
    # PDF workflow requirement: normalize Unicode dash/minus variants.
    body = (
        body.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    yaml = """---
lang: en-GB
reference-section-title: References
link-citations: false
---

"""
    return yaml + body


def _run_pandoc() -> None:
    BODY_MD.write_text(_extract_submission_body())
    subprocess.run(
        [
            str(PANDOC),
            str(BODY_MD),
            "--from=markdown+tex_math_dollars+tex_math_single_backslash",
            "--to=docx",
            "--standalone",
            "--citeproc",
            f"--bibliography={BIBLIOGRAPHY}",
            f"--resource-path={THESIS}",
            f"--output={PANDOC_DOCX}",
        ],
        check=True,
        cwd=ROOT,
    )


def _set_run_font(run, size: float | None = None, bold: bool | None = None) -> None:
    run.font.name = BODY_FONT
    r_fonts = run._element.get_or_add_rPr().rFonts
    for attribute in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        r_fonts.set(qn(attribute), BODY_FONT)
    for attribute in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        r_fonts.attrib.pop(qn(attribute), None)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = INK


def _set_style_font(style, size: float, bold: bool | None = None) -> None:
    style.font.name = BODY_FONT
    r_fonts = style._element.get_or_add_rPr().rFonts
    for attribute in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        r_fonts.set(qn(attribute), BODY_FONT)
    for attribute in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        r_fonts.attrib.pop(qn(attribute), None)
    style.font.size = Pt(size)
    style.font.color.rgb = INK
    if bold is not None:
        style.font.bold = bold


def _configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = _style_by_name(styles, "Normal")
    _set_style_font(normal, BODY_SIZE_PT)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = Pt(BODY_LINE_PT)
    normal.paragraph_format.widow_control = True

    if _has_style(styles, "Body Text"):
        body = _style_by_name(styles, "Body Text")
        _set_style_font(body, BODY_SIZE_PT)
        body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        body.paragraph_format.space_before = Pt(0)
        body.paragraph_format.space_after = Pt(0)
        body.paragraph_format.line_spacing = Pt(BODY_LINE_PT)
        body.paragraph_format.widow_control = True

    heading_tokens = {
        "Heading 1": (16, 18, 10, True),
        "Heading 2": (14, 14, 6, True),
        "Heading 3": (12, 10, 4, True),
    }
    for name, (size, before, after, bold) in heading_tokens.items():
        style = _style_by_name(styles, name)
        _set_style_font(style, size, bold)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        if name == "Heading 1":
            style.paragraph_format.page_break_before = False

    for name in ["Caption", "Image Caption"]:
        if _has_style(styles, name):
            style = _style_by_name(styles, name)
            _set_style_font(style, 10)
            style.font.italic = True
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style.paragraph_format.space_before = Pt(4)
            style.paragraph_format.space_after = Pt(8)
            style.paragraph_format.line_spacing = 1.0
            style.paragraph_format.keep_with_next = False
            style.paragraph_format.keep_together = True

    for name in ["Bibliography", "References"]:
        if _has_style(styles, name):
            style = _style_by_name(styles, name)
            _set_style_font(style, 11)
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            style.paragraph_format.left_indent = Cm(0.75)
            style.paragraph_format.first_line_indent = Cm(-0.75)
            style.paragraph_format.space_after = Pt(5)
            style.paragraph_format.line_spacing = 1.15

    for name in ["TOC Heading", "Contents Heading"]:
        if _has_style(styles, name):
            style = _style_by_name(styles, name)
            _set_style_font(style, 16, True)
            style.paragraph_format.page_break_before = False
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(12)

    for name in ["TOC 1", "TOC 2", "TOC 3"]:
        if _has_style(styles, name):
            style = _style_by_name(styles, name)
            _set_style_font(style, 11)
            style.paragraph_format.line_spacing = 1.15
            style.paragraph_format.space_after = Pt(3)

    # Pandoc may emit compact list styles. Keep them Bocconi-compatible.
    for style in styles:
        if style.type != WD_STYLE_TYPE.PARAGRAPH:
            continue
        if "List" in style.name:
            _set_style_font(style, BODY_SIZE_PT)
            style.paragraph_format.space_after = Pt(0)
            style.paragraph_format.line_spacing = Pt(BODY_LINE_PT)


def _prepend_static_contents(doc: Document) -> None:
    anchor = doc.paragraphs[0]._p
    created = []

    heading = doc.add_paragraph("Contents")
    heading.style = _style_by_name(doc.styles, "TOC Heading")
    heading.paragraph_format.page_break_before = False
    heading.paragraph_format.space_before = Pt(0)
    heading.paragraph_format.space_after = Pt(12)
    created.append(heading)

    for label, page in CONTENTS_ENTRIES:
        paragraph = doc.add_paragraph()
        paragraph.style = _style_by_name(doc.styles, "Normal")
        paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.line_spacing = 1.15
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.paragraph_format.tab_stops.add_tab_stop(
            Cm(15.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
        )
        run = paragraph.add_run(f"{label}\t{page}")
        _set_run_font(run, 11)
        created.append(paragraph)

    for paragraph in created:
        anchor.addprevious(paragraph._p)


def _blank_page_paragraph(with_page_break: bool) -> OxmlElement:
    p = OxmlElement("w:p")
    if with_page_break:
        r = OxmlElement("w:r")
        br = OxmlElement("w:br")
        br.set(qn("w:type"), "page")
        r.append(br)
        p.append(r)
    return p


def _section_break_paragraph(template_sect_pr: OxmlElement) -> OxmlElement:
    p = OxmlElement("w:p")
    p_pr = OxmlElement("w:pPr")
    sect_pr = copy.deepcopy(template_sect_pr)
    for tag in ["w:headerReference", "w:footerReference", "w:pgNumType"]:
        for node in list(sect_pr.findall(qn(tag))):
            sect_pr.remove(node)
    type_node = sect_pr.find(qn("w:type"))
    if type_node is None:
        type_node = OxmlElement("w:type")
        sect_pr.insert(0, type_node)
    type_node.set(qn("w:val"), "nextPage")
    p_pr.append(sect_pr)
    p.append(p_pr)
    return p


def _prepend_bocconi_pages(doc: Document) -> None:
    body = doc._element.body
    final_sect_pr = body.sectPr
    first_content = next(child for child in body if child.tag != qn("w:sectPr"))
    nodes = [
        _blank_page_paragraph(True),
        _blank_page_paragraph(True),
        _blank_page_paragraph(True),
        _blank_page_paragraph(True),
        _section_break_paragraph(final_sect_pr),
    ]
    for node in nodes:
        first_content.addprevious(node)


def _set_page_number_start(section, start: int = 1) -> None:
    sect_pr = section._sectPr
    for node in list(sect_pr.findall(qn("w:pgNumType"))):
        sect_pr.remove(node)
    pg_num = OxmlElement("w:pgNumType")
    pg_num.set(qn("w:start"), str(start))
    sect_pr.append(pg_num)


def _add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    _set_run_font(run, 10)
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char_separate = OxmlElement("w:fldChar")
    fld_char_separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    for node in [fld_char_begin, instr, fld_char_separate, text, fld_char_end]:
        run._r.append(node)


def _configure_sections(doc: Document) -> None:
    if len(doc.sections) != 2:
        raise RuntimeError(f"expected two sections after Bocconi preamble, got {len(doc.sections)}")
    for section in doc.sections:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Cm(PAGE_WIDTH_CM)
        section.page_height = Cm(PAGE_HEIGHT_CM)
        section.left_margin = Cm(MARGIN_SIDE_CM)
        section.right_margin = Cm(MARGIN_SIDE_CM)
        section.top_margin = Cm(MARGIN_TOP_CM)
        section.bottom_margin = Cm(MARGIN_BOTTOM_CM)
        section.header_distance = Cm(1.25)
        section.footer_distance = Cm(1.25)
        section.different_first_page_header_footer = False

    blank, content = doc.sections
    blank.start_type = WD_SECTION_START.NEW_PAGE
    blank.header.is_linked_to_previous = False
    blank.footer.is_linked_to_previous = False
    for paragraph in list(blank.header.paragraphs) + list(blank.footer.paragraphs):
        paragraph.clear()

    content.start_type = WD_SECTION_START.NEW_PAGE
    content.header.is_linked_to_previous = False
    content.footer.is_linked_to_previous = False
    for paragraph in content.header.paragraphs:
        paragraph.clear()
    footer = content.footer
    for paragraph in footer.paragraphs:
        paragraph.clear()
    _add_page_field(footer.paragraphs[0])
    _set_page_number_start(content, 1)


def _set_cell_margins(cell, top=70, start=90, bottom=70, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def _prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def _set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), "4")
        tag.set(qn("w:color"), "B7B7B7")


def _column_widths(table) -> list[int]:
    cols = len(table.columns)
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    if cols == 2:
        weights = [0.58, 0.42] if len(headers[0]) > len(headers[1]) else [0.42, 0.58]
    elif cols == 3:
        if headers[0] == "Database or provider":
            # Give the provider-supplied information column enough room for
            # terms such as "recommendations" and "implied volatility".
            weights = [0.29, 0.27, 0.44]
        else:
            weights = [0.34, 0.16, 0.50]
    elif cols == 4:
        weights = [0.38, 0.21, 0.21, 0.20]
    elif cols == 5:
        weights = [0.22, 0.18, 0.20, 0.20, 0.20]
    elif cols == 7:
        weights = [0.17, 0.28, 0.13, 0.105, 0.105, 0.105, 0.105]
    else:
        weights = [1.0 / cols] * cols
    widths = [int(USABLE_WIDTH_DXA * weight) for weight in weights]
    widths[-1] += USABLE_WIDTH_DXA - sum(widths)
    return widths


def _configure_tables(doc: Document) -> None:
    for table in doc.tables:
        # Pandoc 3 can emit a valid simple table with an empty ``tblGrid``.
        # Word renders it, but python-docx then reports zero columns.  Restore
        # the grid from the physical cells before applying explicit widths.
        if len(table.columns) == 0:
            physical_cols = max(len(row._tr.tc_lst) for row in table.rows)
            provisional = USABLE_WIDTH_DXA // physical_cols
            for _ in range(physical_cols):
                grid_col = OxmlElement("w:gridCol")
                grid_col.set(qn("w:w"), str(provisional))
                table._tbl.tblGrid.append(grid_col)

        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        table.autofit = False
        tbl_pr = table._tbl.tblPr
        tbl_w = tbl_pr.first_child_found_in("w:tblW")
        if tbl_w is None:
            tbl_w = OxmlElement("w:tblW")
            tbl_pr.append(tbl_w)
        tbl_w.set(qn("w:w"), str(USABLE_WIDTH_DXA))
        tbl_w.set(qn("w:type"), "dxa")
        tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
        if tbl_ind is None:
            tbl_ind = OxmlElement("w:tblInd")
            tbl_pr.append(tbl_ind)
        tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
        tbl_ind.set(qn("w:type"), "dxa")
        layout = tbl_pr.first_child_found_in("w:tblLayout")
        if layout is None:
            layout = OxmlElement("w:tblLayout")
            tbl_pr.append(layout)
        layout.set(qn("w:type"), "fixed")
        _set_table_borders(table)

        widths = _column_widths(table)
        grid = table._tbl.tblGrid
        for node in list(grid):
            grid.remove(node)
        for width in widths:
            grid_col = OxmlElement("w:gridCol")
            grid_col.set(qn("w:w"), str(width))
            grid.append(grid_col)

        _set_repeat_table_header(table.rows[0])
        for r_idx, row in enumerate(table.rows):
            _prevent_row_split(row)
            for c_idx, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.width = Inches(widths[c_idx] / 1440)
                tc_pr = cell._tc.get_or_add_tcPr()
                tc_w = tc_pr.first_child_found_in("w:tcW")
                tc_w.set(qn("w:w"), str(widths[c_idx]))
                tc_w.set(qn("w:type"), "dxa")
                _set_cell_margins(cell)
                if r_idx == 0:
                    shd = tc_pr.first_child_found_in("w:shd")
                    if shd is None:
                        shd = OxmlElement("w:shd")
                        tc_pr.append(shd)
                    shd.set(qn("w:fill"), LIGHT_GRAY)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.alignment = (
                        WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    )
                    for run in paragraph.runs:
                        _set_run_font(run, 8.5 if len(widths) >= 5 else 9.5, r_idx == 0)


def _configure_paragraphs_and_images(doc: Document) -> None:
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        has_math = bool(paragraph._p.xpath(".//m:oMath | .//m:oMathPara"))
        has_drawing = bool(paragraph._p.xpath(".//w:drawing"))
        if text == "Table of Contents":
            paragraph.text = "Contents"
            paragraph.style = (
                _style_by_name(doc.styles, "TOC Heading")
                if _has_style(doc.styles, "TOC Heading")
                else _style_by_name(doc.styles, "Heading 1")
            )
            paragraph.paragraph_format.page_break_before = False
        if paragraph.style.name == "Heading 1":
            paragraph.paragraph_format.page_break_before = text in {
                "1. Introduction",
                "References",
            }
        if has_drawing:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.keep_together = True
            paragraph.paragraph_format.keep_with_next = True
        elif has_math and not text:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(4)
        elif "\t" in paragraph.text:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif paragraph.style.name not in {"Heading 1", "Heading 2", "Heading 3", "Caption", "Image Caption", "Bibliography", "References", "TOC Heading", "TOC 1", "TOC 2", "TOC 3"}:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in paragraph.runs:
            if run._r.xpath(".//m:oMath") or run._r.xpath(".//m:oMathPara"):
                continue
            size = BODY_SIZE_PT
            if paragraph.style.name == "Heading 1":
                size = 16
            elif paragraph.style.name == "Heading 2":
                size = 14
            elif paragraph.style.name == "Heading 3":
                size = 12
            elif paragraph.style.name in {"Caption", "Image Caption"}:
                size = 10
            elif paragraph.style.name.startswith("TOC"):
                size = 11
            elif paragraph.style.name in {"Bibliography", "References"}:
                size = 11
            _set_run_font(run, size)

    # Give dense statistical charts more width while keeping the chronology
    # compact enough to avoid an orphaned continuation page.
    figure_widths_cm = [13.8, 13.0, 11.5, 13.4, 13.8, 13.4, 13.8, 13.8]
    for index, shape in enumerate(doc.inline_shapes):
        target = Cm(
            figure_widths_cm[index]
            if index < len(figure_widths_cm)
            else figure_widths_cm[-1]
        )
        if shape.width > target:
            ratio = target / shape.width
            shape.width = target
            shape.height = int(shape.height * ratio)


def _clear_personal_metadata(doc: Document) -> None:
    props = doc.core_properties
    props.author = ""
    props.last_modified_by = ""
    props.comments = ""
    props.keywords = "machine learning; asset pricing; stock selection; long-short portfolio"
    props.subject = "Bachelor Final Paper"
    props.title = "Machine Learning in Portfolio Construction: Stock Selection and Dynamic Equity Allocation"


def _update_fields_on_open(doc: Document) -> None:
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    _run_pandoc()
    doc = Document(PANDOC_DOCX)
    _prepend_static_contents(doc)
    _configure_styles(doc)
    _prepend_bocconi_pages(doc)
    temp = HERE / "_with_sections.docx"
    doc.save(temp)

    # Reopen so python-docx discovers both OOXML sections.
    doc = Document(temp)
    _configure_styles(doc)
    _configure_sections(doc)
    _configure_paragraphs_and_images(doc)
    _configure_tables(doc)
    _clear_personal_metadata(doc)
    _update_fields_on_open(doc)
    doc.save(OUTPUT_DOCX)
    PANDOC_DOCX.unlink(missing_ok=True)
    temp.unlink(missing_ok=True)
    print(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
