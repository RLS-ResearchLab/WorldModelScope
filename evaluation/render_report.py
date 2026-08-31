"""Render results/REPORT.md to results/REPORT.pdf (Unicode-safe, DejaVu fonts).

Block-level Markdown only: #..#### headings, paragraphs with **bold** / *italic* /
`code`, ``` fenced blocks, | tables |, - lists, --- rules. Enough for the report.

    .venv/bin/python evaluation/render_report.py
"""
import glob
import re
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

SRC = Path(__file__).with_name("..") / "results" / "REPORT.md"
SRC = (Path(__file__).resolve().parent.parent / "results" / "REPORT.md")
PDF_OUT = SRC.with_suffix(".pdf")


def _find(*names):
    for n in names:
        for root in ("/usr/share/fonts", "/snap", str(Path.home() / ".cache")):
            hits = glob.glob(f"{root}/**/{n}", recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(names)


SANS = _find("DejaVuSans.ttf")
SANS_B = _find("DejaVuSans-Bold.ttf")
SANS_I = _find("DejaVuSans-Oblique.ttf")
MONO = _find("DejaVuSansMono.ttf")
SERIF_B = _find("DejaVuSerif-Bold.ttf")

INK = (24, 28, 36)
ACCENT = (150, 78, 23)
SOFT = (90, 100, 115)
RULE = (210, 214, 222)
CODE_BG = (244, 244, 246)

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_CODE = re.compile(r"`([^`]+)`")


class Doc(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-13)
        self.set_font("sans", "", 7.5)
        self.set_text_color(160)
        self.cell(0, 8, f"WorldModelScope · evaluation report · {self.page_no()}", align="C")


def _write_rich(pdf: Doc, text: str, size=10, lead=5.0, color=INK):
    """One paragraph with inline **bold** / *italic* / `code`."""
    pdf.set_text_color(*color)
    # tokenise into (style, chunk)
    tokens = []
    i = 0
    pat = re.compile(r"\*\*(.+?)\*\*|(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)|`([^`]+)`")
    for mobj in pat.finditer(text):
        if mobj.start() > i:
            tokens.append(("", text[i:mobj.start()]))
        if mobj.group(1) is not None:
            tokens.append(("B", mobj.group(1)))
        elif mobj.group(2) is not None:
            tokens.append(("I", mobj.group(2)))
        else:
            tokens.append(("C", mobj.group(3)))
        i = mobj.end()
    if i < len(text):
        tokens.append(("", text[i:]))
    if not tokens:
        tokens = [("", text)]

    for style, chunk in tokens:
        for j, word in enumerate(re.split(r"(\s+)", chunk)):
            if word == "":
                continue
            if style == "C":
                pdf.set_font("mono", "", size - 1)
            elif style == "B":
                pdf.set_font("sans", "B", size)
            elif style == "I":
                pdf.set_font("sans", "I", size)
            else:
                pdf.set_font("sans", "", size)
            w = pdf.get_string_width(word)
            if word.strip() == "" and pdf.get_x() <= pdf.l_margin + 0.1:
                continue
            if pdf.get_x() + w > pdf.w - pdf.r_margin:
                pdf.ln(lead)
            pdf.cell(w, lead, word)
    pdf.ln(lead)


def render():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    pdf = Doc(format="A4")
    for fam, path in [("sans", SANS), ("mono", MONO), ("serif", SERIF_B)]:
        pdf.add_font(fam, "", path)
    pdf.add_font("sans", "B", SANS_B)
    pdf.add_font("sans", "I", SANS_I)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(17, 15, 17)
    pdf.add_page()
    EPW = pdf.epw

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.rstrip()

        # fenced code
        if s.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            pdf.ln(1)
            pdf.set_font("mono", "", 8)
            pdf.set_fill_color(*CODE_BG)
            pdf.set_text_color(*INK)
            for b in block:
                pdf.multi_cell(EPW, 4.4, b or " ", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            continue

        # table
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            grid = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            grid = [r for r in grid if not all(set(c) <= set("-: ") for c in r)]
            if grid:
                pdf.ln(1)
                pdf.set_font("sans", "", 8)
                with pdf.table(borders_layout="MINIMAL", line_height=5,
                               text_align="LEFT", first_row_as_headings=True,
                               headings_style=FontFace(emphasis="BOLD", fill_color=(238, 240, 244),
                                                       color=INK)) as table:
                    for gi, row in enumerate(grid):
                        r = table.row()
                        for cell in row:
                            c = _CODE.sub(r"\1", _BOLD.sub(r"\1", cell))
                            c = re.sub(r"</?sub>|</?br\s*/?>", "", c)
                            c = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"\1", c)
                            r.cell(c)
                pdf.ln(2)
            continue

        if not s:
            pdf.ln(2.2)
            i += 1
            continue

        if s == "---":
            pdf.ln(1)
            pdf.set_draw_color(*RULE)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(3)
            i += 1
            continue

        if s.startswith("#### "):
            pdf.ln(3)
            pdf.set_font("sans", "B", 10.5)
            pdf.set_text_color(*ACCENT)
            pdf.multi_cell(EPW, 5.5, _BOLD.sub(r"\1", s[5:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif s.startswith("### "):
            pdf.ln(4)
            pdf.set_font("serif", "", 12.5)
            pdf.set_text_color(*INK)
            pdf.multi_cell(EPW, 6.5, s[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif s.startswith("## "):
            pdf.ln(6)
            pdf.set_draw_color(*ACCENT)
            pdf.set_line_width(0.5)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.l_margin + 16, y)
            pdf.ln(2)
            pdf.set_font("serif", "", 15)
            pdf.set_text_color(*INK)
            pdf.multi_cell(EPW, 7.5, s[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        elif s.startswith("# "):
            pdf.ln(2)
            pdf.set_font("serif", "", 20)
            pdf.set_text_color(*INK)
            pdf.multi_cell(EPW, 9, s[2:], new_x="LMARGIN", new_y="NEXT", align="L")
            pdf.ln(3)
        elif re.match(r"^\s*(?:[-*]|\d+\.) ", s):
            body = [re.sub(r"^\s*(?:[-*]|\d+\.) ", "", s)]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and lines[i + 1].startswith(("  ", "\t"))
                   and not re.match(r"^\s*(?:[-*]|\d+\.) ", lines[i + 1])):
                i += 1
                body.append(lines[i].strip())
            indent = 4
            pdf.set_x(pdf.l_margin + indent)
            pdf.set_font("sans", "", 10)
            pdf.set_text_color(*ACCENT)
            marker = "•" if re.match(r"^\s*[-*] ", s) else s.strip().split(".")[0] + "."
            pdf.cell(5.5, 5, marker)
            pdf.set_text_color(*INK)
            start_x = pdf.get_x()
            pdf.set_x(start_x)
            _write_rich_indent(pdf, " ".join(body), start_x)
        elif s.startswith("> "):
            _write_rich(pdf, s[2:], size=9.5, lead=4.6, color=SOFT)
        else:
            # accumulate a wrapped paragraph
            para = [s]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not lines[i + 1].lstrip().startswith(("#", "|", "```", "- ", "* ", ">", "---"))
                   and not re.match(r"^\s*\d+\. ", lines[i + 1])):
                i += 1
                para.append(lines[i].strip())
            _write_rich(pdf, " ".join(para), size=10, lead=5.0)
        i += 1

    pdf.output(str(PDF_OUT))
    kb = PDF_OUT.stat().st_size // 1024
    print(f"wrote {PDF_OUT}  ({pdf.page_no()} pages, {kb} KB)")


def _write_rich_indent(pdf, text, start_x):
    saved_l = pdf.l_margin
    pdf.set_left_margin(start_x)
    pdf.set_x(start_x)
    _write_rich(pdf, text, size=10, lead=5.0)
    pdf.set_left_margin(saved_l)


if __name__ == "__main__":
    render()
