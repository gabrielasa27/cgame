"""Genera client/index.html inyectando los niveles compartidos y el modelo SVM entrenado.

Uso (desde la raíz del proyecto):  python tools/build_client.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
template = (ROOT / "client" / "src" / "index.template.html").read_text(encoding="utf-8")
levels = json.loads((ROOT / "shared" / "levels.json").read_text(encoding="utf-8"))
model = json.loads((ROOT / "backend" / "data" / "svm_model.json").read_text(encoding="utf-8"))

for marker, payload in (("/*__LEVELS_JSON__*/null", levels), ("/*__SVM_MODEL__*/null", model)):
    assert template.count(marker) == 1, f"Falta el marcador {marker}"
    # "</" se escapa para que el JSON nunca cierre la etiqueta <script>
    template = template.replace(marker, json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))

out = ROOT / "client" / "index.html"
out.write_text(template, encoding="utf-8")
print("Generado", out, f"({out.stat().st_size / 1024:.0f} KB)")
