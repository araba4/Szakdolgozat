import math


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def sci(v):
    if _missing(v):
        return "–"
    if v == 0:
        return "0"
    mant, exp = f"{v:.2e}".split("e")
    return f"{mant.replace('-', '−')}e{int(exp)}"


def f(v, nd=2):
    return "–" if _missing(v) else f"{v:.{nd}f}".replace("-", "−")


def pct(v, nd=2):
    return "–" if _missing(v) else f"{100 * v:.{nd}f}%".replace("-", "−")


def chg(v, nd=1):
    return "–" if _missing(v) else f"{v:+.{nd}f}%".replace("-", "−")


def p(n):
    return "–" if _missing(n) else f"{int(n):,}".replace(",", " ")


def yes(b):
    return "–" if b is None else ("igen" if b else "nem")


def table(header, rows, align=None):
    if align is None:
        align = "l" + "r" * (len(header) - 1)
    sep = "|" + "|".join("---" if a == "l" else "---:" for a in align) + "|"
    out = ["| " + " | ".join(header) + " |", sep]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out
