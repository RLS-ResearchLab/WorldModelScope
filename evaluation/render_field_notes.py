"""Render evaluation/FIELD_NOTES.md to a standalone PDF + plain-text file in the
user's home dir (outside the git repo). Block-by-block renderer -- no HTML, so
wide tables and inline code never break it."""
import re
from pathlib import Path

from fpdf import FPDF

SRC = Path(__file__).with_name("FIELD_NOTES.md")
HOME = Path.home()
PDF_OUT = SRC.with_suffix(".pdf")
TXT_OUT = SRC.with_suffix(".txt")

lines = SRC.read_text(encoding="utf-8").splitlines()

_UNI = {
    "≈": "~", "≥": ">=", "≤": "<=", "≫": ">>", "→": "->", "×": "x", "÷": "/",
    "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "·": "*",
    "…": "...", "²": "2", "τ": "tau", "√": "sqrt", "≠": "!=", "⇒": "=>",
    "▪": "-", "●": "o", "─": "-", "•": "-", "−": "-", "⁻": "-", "₁": "1",
    "₂": "2", "₊": "+", "±": "+/-", "°": "deg", "‑": "-", "‖": "||", "∥": "||",
    " ": " ", " ": " ", " ": " ", "‑": "-",
}
_INLINE = re.compile(r"\*\*(.+?)\*\*")
_TICK = re.compile(r"`([^`]+)`")


def strip_md(s: str) -> str:
    return _TICK.sub(r"\1", _INLINE.sub(r"\1", s))


def clean(s: str) -> str:
    s = strip_md(s)
    for k, v in _UNI.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def realign_table(rows):
    """rows: list of raw '| a | b |' strings. Returns aligned monospace lines."""
    grid = [[strip_md(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows]
    grid = [r for r in grid if not all(set(c) <= set("-: ") for c in r)]  # drop |---|
    if not grid:
        return []
    ncol = max(len(r) for r in grid)
    grid = [r + [""] * (ncol - len(r)) for r in grid]
    w = [max(len(row[i]) for row in grid) for i in range(ncol)]
    out = []
    for j, row in enumerate(grid):
        out.append("  ".join(row[i].ljust(w[i]) for i in range(ncol)).rstrip())
        if j == 0:
            out.append("  ".join("-" * w[i] for i in range(ncol)))
    return out


# ---------------- plain text ----------------
txt = []
buf = []
for ln in lines:
    if ln.startswith("|"):
        buf.append(ln)
        continue
    if buf:
        txt += realign_table(buf)
        buf = []
    txt.append(re.sub(r"\*\*(.+?)\*\*", r"\1", re.sub(r"`([^`]+)`", r"\1", ln)).replace("```", ""))
if buf:
    txt += realign_table(buf)
TXT_OUT.write_text("\n".join(txt) + "\n", encoding="utf-8")


# ---------------- PDF ----------------
class Doc(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"WorldModelScope eval status   -   {self.page_no()}", align="C")


pdf = Doc(format="A4")
pdf.set_auto_page_break(auto=True, margin=16)
pdf.set_margins(16, 15, 16)
pdf.add_page()
EPW = pdf.epw

INK = (26, 34, 46)
ACC = (150, 78, 23)
SOFT = (90, 100, 115)


def para(text, size=10, style="", lead=5, gap=1.5, color=INK, indent=0):
    text = clean(re.sub(r"\*\*(.+?)\*\*", r"\1", re.sub(r"`([^`]+)`", r"\1", text)))
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    if indent:
        pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(EPW - indent, lead, text, new_x="LMARGIN", new_y="NEXT")
    if gap:
        pdf.ln(gap)


def mono_block(rows, size=7.5):
    pdf.set_font("Courier", "", size)
    pdf.set_text_color(*INK)
    pdf.set_fill_color(244, 244, 246)
    for r in rows:
        pdf.multi_cell(EPW, size * 0.52 + 1.3, clean(r) or " ", new_x="LMARGIN",
                       new_y="NEXT", fill=True)
    pdf.ln(2)


i = 0
tbl = []
in_code = False
code = []
while i < len(lines):
    ln = lines[i]

    if ln.strip().startswith("```"):
        if in_code:
            mono_block(code)
            code = []
        in_code = not in_code
        i += 1
        continue
    if in_code:
        code.append(ln)
        i += 1
        continue

    if ln.startswith("|"):
        tbl.append(ln)
        i += 1
        continue
    if tbl:
        mono_block(realign_table(tbl))
        tbl = []

    s = ln.strip()
    if not s:
        pdf.ln(2)
    elif s.startswith("# "):
        pdf.ln(3); para(s[2:], size=17, style="B", lead=8, gap=2, color=INK)
    elif s.startswith("## "):
        pdf.ln(4)
        pdf.set_draw_color(*ACC); pdf.set_line_width(0.5)
        y = pdf.get_y(); pdf.line(pdf.l_margin, y, pdf.l_margin + 14, y); pdf.ln(2)
        para(s[3:], size=13.5, style="B", lead=6.5, gap=2.5, color=INK)
    elif s.startswith("### "):
        pdf.ln(2); para(s[4:], size=11, style="B", lead=5.5, gap=1.5, color=ACC)
    elif s.startswith(">"):
        para(s.lstrip("> ").strip() or " ", size=9.5, style="I", lead=4.6,
             color=SOFT, indent=5)
    elif re.match(r"^[-*] ", s):
        para("-  " + s[2:], size=10, lead=4.8, gap=1, indent=4)
    elif re.match(r"^\d+\. ", s):
        para(s, size=10, lead=4.8, gap=1, indent=4)
    elif s == "---":
        pdf.ln(1)
    else:
        para(s)
    i += 1

if tbl:
    mono_block(realign_table(tbl))

pdf.output(str(PDF_OUT))
print("wrote", PDF_OUT, f"({PDF_OUT.stat().st_size // 1024} KB)")
print("wrote", TXT_OUT, f"({TXT_OUT.stat().st_size // 1024} KB)")
