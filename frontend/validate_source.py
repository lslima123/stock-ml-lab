import json
from pathlib import Path

root = Path(__file__).resolve().parent
package = json.loads((root / "package.json").read_text())
assert package["version"] == "0.15.0"
assert package["dependencies"]["react"].startswith("19.")
assert package["scripts"]["build"]
assert (root / "src/App.tsx").exists()
assert (root / "src/lib/api.ts").exists()

app = (root / "src/App.tsx").read_text()
api = (root / "src/lib/api.ts").read_text()
formatters = (root / "src/lib/format.ts").read_text()
vite = (root / "vite.config.ts").read_text()

for scope in ("local", "global", "compare"):
    assert scope in app
assert "/api/v1/capabilities" in api
assert "/api/v1/predict" in api
assert "/api/v1/research/summary" in api
assert 'base: "/app/"' in vite
assert '"/docs":' in vite
assert '"/openapi.json":' in vite
assert "formatPctPoint(prediction.comparison.global_minus_local)" in app
assert 'p.p.`' in formatters
assert "formatPercentValue" in formatters
assert "scope_benchmark" in (root / "src/types.ts").read_text()
assert "scope_confirmation" in (root / "src/types.ts").read_text()
assert "M14" in app
assert "M15" in app
evidence = (root / "src/components/ResearchEvidence.tsx").read_text()
assert "post hoc" in evidence
assert "Not confirmed" in evidence
assert "forecasting-loss evidence" in evidence
assert "When M12 turns the global scope on" not in app

print("Front-end source contract: OK")

assert (root / "src/vite-env.d.ts").exists()
print("Vite TypeScript declarations: OK")
