# --- WSGI export e healthcheck (Cloud Run) ---
try:
    server = app.server
except Exception:
    pass

try:
    @server.get("/healthz")
    def _healthz():
        return "ok", 200
except Exception:
    pass
