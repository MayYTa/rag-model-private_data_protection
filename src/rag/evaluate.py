"""Mide la recuperación en dev. Uso: python -m rag.evaluate [--stem]"""
import json
import sys
from pathlib import Path

from rag.retrieve import BM25, load_corpus


def first_hit(results: list[dict], gold: dict) -> int | None:
    """Posición (desde 1) del artículo principal, el primero de la lista."""
    article = gold["articulos"][0].split(".")[0]
    for position, row in enumerate(results, start=1):
        if row["articulo"] == article and gold["norma"].startswith(row["norma"]):
            return position
    return None


def main(split: str = "dev", stem: bool = False) -> None:
    index = BM25(load_corpus(), stem=stem)
    lines = Path("eval/eval_set.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [q for q in map(json.loads, lines) if q["split"] == split and q["articulos"]]
    ranks = []
    for q in questions:
        results = index.search(q["pregunta"], k=10)
        rank = first_hit(results, q)
        ranks.append(rank)
        top = ", ".join(f"{r['norma'].split()[0]}:{r['articulo']}" for r in results[:5])
        print(f"{q['id']} esperado={q['articulos'][0]:<5} posición={rank}  top5: {top}")
    n = len(ranks)
    for k in (1, 5, 10):
        print(f"recall@{k}: {sum(r is not None and r <= k for r in ranks) / n:.2f}")
    print(f"MRR@10: {sum(1 / r for r in ranks if r) / n:.2f}  (n={n})")


if __name__ == "__main__":
    main(stem="--stem" in sys.argv)
