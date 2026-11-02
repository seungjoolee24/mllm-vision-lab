"""Validate config, notebook code, imported evidence links, and accidental secrets."""
import ast
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visionlab.core import validate_config

for path in (ROOT / "configs").glob("*.json"):
    validate_config(json.loads(path.read_text(encoding="utf-8")))
for path in (ROOT / "notebooks").glob("*.ipynb"):
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["nbformat"] == 4
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    # Check the nested askpass program actually has newlines and valid Python.
    source = "".join(next(c for c in value["cells"] if c["cell_type"] == "code" and "askpass.write_text" in "".join(c["source"]))["source"])
    tree = ast.parse(source)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "write_text" and isinstance(n.args[0], ast.Constant))
    ast.parse(call.args[0].value)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src"} and value and not value.startswith(("http", "#", "data:")):
                self.links.append(value)

for path in (ROOT / "history").rglob("*.html"):
    parser = Links()
    parser.feed(path.read_text(encoding="utf-8"))
    assert all((path.parent / ref.split("#", 1)[0]).is_file() for ref in parser.links), path

secret_pattern = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
for directory in ("visionlab", "scripts", "configs", "notebooks", "manifests", "docs", ".github"):
    for path in (ROOT / directory).rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".ipynb", ".md", ".yml"}:
            assert not secret_pattern.search(path.read_text(encoding="utf-8")), f"Credential-like value in {path}"
print("Validated configurations, clean notebook Python, historical report links and credential scan.")
