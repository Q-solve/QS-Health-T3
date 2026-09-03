"""
embed_figures.py
================
Replace <img data-fig="figures/x.png"> placeholders in an HTML file with
self-contained base64 data URIs, so the published page carries its own images.

    python embed_figures.py briefing_src.html briefing.html
"""

import base64
import os
import re
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    html = open(src, encoding="utf-8").read()

    total = 0

    def repl(m):
        nonlocal total
        path = m.group(1)
        if not os.path.exists(path):
            print(f"  MISSING {path}")
            return m.group(0)
        raw = open(path, "rb").read()
        total += len(raw)
        b64 = base64.b64encode(raw).decode("ascii")
        print(f"  embedded {path}  ({len(raw)/1024:.0f} KB)")
        return f'src="data:image/png;base64,{b64}"'

    html = re.sub(r'data-fig="([^"]+)"', repl, html)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\n{len(html)/1024/1024:.2f} MB written to {dst} "
          f"({total/1024/1024:.2f} MB of images)")


if __name__ == "__main__":
    main()
