import os, importlib.util, sys

BASE = os.path.dirname(__file__)

PREFER = ["app.py", "MULTIMERCADO.py", "etapa2_dash_mm.py"]
cands  = [p for p in PREFER if os.path.exists(os.path.join(BASE, p))]
cands += [f for f in os.listdir(BASE) if f.endswith(".py") and f not in cands and f not in ("wsgi.py",)]

def load_mod(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

for fn in cands:
    p = os.path.join(BASE, fn)
    try:
        m = load_mod(p)
        app = getattr(m, "app", None)
        if app is not None and hasattr(app, "server"):
            server = app.server
            # print(f"WSGI usando: {fn}")
            break
    except Exception:
        continue
else:
    raise RuntimeError("Nenhum módulo com 'app.server' encontrado. Garanta que seu código tenha 'app = Dash(...)'.")

