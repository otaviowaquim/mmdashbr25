import json, sys, io, re, os

ip, op = sys.argv[1], sys.argv[2]
nb = json.load(open(ip, encoding="utf-8"))
buf = io.StringIO()

def clean(src: str) -> str:
    out = []
    for ln in src.splitlines(True):
        s = ln.lstrip()
        if s.startswith("%") or s.startswith("!") or s.startswith("??") or "get_ipython(" in s:
            continue
        out.append(ln.replace("\t", "    "))
    txt = "".join(out)
    if not txt.endswith("\n"):
        txt += "\n"
    return txt

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        buf.write(clean("".join(cell.get("source", []))))
        buf.write("\n# ----\n\n")

code = buf.getvalue()
if "server = app.server" not in code:
    code += "\n# WSGI export (se existir app)\ntry:\n    server = app.server\nexcept Exception:\n    pass\n"

open(op, "w", encoding="utf-8").write(code)
