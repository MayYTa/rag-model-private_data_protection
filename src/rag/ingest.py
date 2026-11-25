"""Parte cada norma en un registro por artículo y revisa la numeración.

Uso: python -m rag.ingest   (desde la raíz del repo)
"""
import json
import re
from collections import Counter
from pathlib import Path

DOCS = [  # (archivo, norma, salida, último artículo esperado)
    ("data/raw/ley_29733.pdf", "Ley 29733", "ley.jsonl", 40),
    ("data/raw/reglamento_ds_016_2024.pdf", "DS 016-2024-JUS", "reglamento.jsonl", 135),
]

# La fuente del PDF de la ley codifica las ligaduras con caracteres privados.
LIGATURES = str.maketrans({"": "fi", "": "fi", "": "fl"})
NOISE = re.compile(r"^[ \t]*(NORMAS LEGALES.*|NORMAS|LEGALES)[ \t]*$", re.MULTILINE)
# Los artículos modificados empiezan con comillas: “Artículo 3.
HEAD = re.compile(r"^[ \t]*[“\"]?Artículo\s+(\d+|[IVXL]+)\s*[°º]?\s*\.", re.MULTILINE)
START = re.compile(r"^[ \t]*TÍTULO PRELIMINAR", re.MULTILINE)
END = re.compile(r"^[ \t]*DISPOSICI(?:ÓN|ONES) COMPLEMENTARIAS?\s+FINAL", re.MULTILINE)
SECTION = re.compile(r"^[ \t]*(TÍTULO|CAPÍTULO|SUB ?CAPÍTULO)\b", re.MULTILINE)


def read_pdf(path: str) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        pages = []
        for number, page in enumerate(doc, start=1):
            lines = page.get_text().splitlines()
            # quita la línea que es solo el número de página
            pages.append("\n".join(x for x in lines if x.strip() != str(number)))
        return "\n".join(pages)


def clean(text: str) -> str:
    return NOISE.sub("", text.translate(LIGATURES))


def split_articles(text: str, norma: str) -> list[dict]:
    """Solo el articulado: del Título Preliminar a las disposiciones complementarias."""
    start = START.search(text)
    if start:  # descarta carátula, considerandos y artículos del decreto aprobatorio
        text = text[start.end():]
    end = END.search(text)
    if end:
        text = text[:end.start()]
    heads = list(HEAD.finditer(text))
    limits = [h.start() for h in heads[1:]] + [len(text)]
    rows = []
    for head, limit in zip(heads, limits):
        body = text[head.start():limit]
        section = SECTION.search(body)
        if section:
            body = body[:section.start()]
        rows.append({"norma": norma, "articulo": head.group(1), "texto": " ".join(body.split())})
    return rows


def report(rows: list[dict], expected: int) -> None:
    nums = [int(r["articulo"]) for r in rows if r["articulo"].isdigit()]
    repeated = sorted(a for a, n in Counter(r["articulo"] for r in rows).items() if n > 1)
    sizes = sorted((len(r["texto"]), r["articulo"]) for r in rows)
    print(f"{rows[0]['norma'] if rows else '?'}: {len(rows)} registros")
    print(f"  repetidos: {repeated}")
    print(f"  faltantes: {sorted(set(range(1, expected + 1)) - set(nums))}")
    print(f"  más cortos: {sizes[:3]}")
    print(f"  más largos: {sizes[-3:]}")


def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for source, norma, name, expected in DOCS:
        rows = split_articles(clean(read_pdf(source)), norma)
        lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in rows]
        (out_dir / name).write_text("".join(lines), encoding="utf-8")
        report(rows, expected)


if __name__ == "__main__":
    main()
