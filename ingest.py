"""Parte una norma en un registro por artículo y revisa la numeración.

Uso: python ingest.py ARCHIVO(.pdf|.txt) NOMBRE_NORMA SALIDA.jsonl
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Encabezado al inicio de línea: "Artículo 5.-", "Artículo 69.", "Artículo III."
HEAD = re.compile(r"^[ \t]*Art[ií]culo[ \t]+(\d+|[IVXL]+)[ \t]*[°º]?[ \t]*\.[ \t]*-?",
                  re.MULTILINE | re.IGNORECASE)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def split_articles(text: str, norma: str) -> list[dict]:
    heads = list(HEAD.finditer(text))
    ends = [h.start() for h in heads[1:]] + [len(text)]
    return [{"norma": norma, "articulo": h.group(1).upper(),
             "texto": text[h.start():end].strip()} for h, end in zip(heads, ends)]


def report(rows: list[dict]) -> None:
    nums = [int(r["articulo"]) for r in rows if r["articulo"].isdigit()]
    repeated = sorted(a for a, n in Counter(r["articulo"] for r in rows).items() if n > 1)
    missing = sorted(set(range(1, max(nums, default=0) + 1)) - set(nums))
    sizes = sorted((len(r["texto"]), r["articulo"]) for r in rows)
    print(f"Artículos: {len(rows)} (máximo: {max(nums, default=0)})")
    print(f"Repetidos: {repeated}")
    print(f"Faltantes: {missing}")
    if sizes:
        print(f"Más cortos (caracteres, artículo): {sizes[:3]}")
        print(f"Más largos (caracteres, artículo): {sizes[-3:]}")


if __name__ == "__main__":
    source, norma, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    rows = split_articles(read_text(source), norma)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    report(rows)
