"""Export docs/report.md to docs/report.pdf (Markdown → HTML → headless Chrome print).

  python scripts/export_pdf.py [--chrome PATH]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c2321; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; } th, td { border: 1px solid #ccc; padding: 2pt 5pt; text-align: left; }
th { background: #f2f1ec; } code { font: 8.5pt ui-monospace, Menlo, monospace; background: #f2f1ec; padding: 0 2pt; }
pre { background: #f2f1ec; padding: 6pt; font-size: 8pt; white-space: pre-wrap; } pre code { background: none; }
img { max-width: 100%; } blockquote { color: #555; border-left: 3px solid #ccc; margin: 6pt 0; padding-left: 8pt; }
a { color: #146356; }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    args = ap.parse_args()
    md = (DOCS / "report.md").read_text()
    # python-markdown needs 4-space nesting (GitHub accepts 2)
    md = "\n".join(("  " + l) if l.startswith("  - ") else l for l in md.splitlines())
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
    html = DOCS / "_report.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>HomePulse Vision report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    out = DOCS / "report.pdf"
    subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", html.as_uri()], check=True, capture_output=True)
    html.unlink()
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
