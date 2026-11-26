"""Mide la recuperación en dev. Uso: python -m rag.evaluate [--method bm25|dense|hybrid] [--stem]"""
import argparse
import json
from pathlib import Path

from rag.dense import Dense, Hybrid
from rag.retrieve import BM25, load_corpus


def first_hit(results: list[dict], gold: dict) -> int | None:
    """Posición (desde 1) del artículo principal, el primero de la lista."""
    article = gold["articulos"][0].split(".")[0]
    for position, row in enumerate(results, start=1):
        if row["articulo"] == article and gold["norma"].startswith(row["norma"]):
            return position
    return None


def build(method: str, stem: bool):
    rows = load_corpus()
    if method == "bm25":
        return BM25(rows, stem=stem)
    if method == "dense":
        return Dense(rows)
    return Hybrid(rows, [BM25(rows, stem=stem), Dense(rows)])


def main(method: str = "bm25", stem: bool = False, split: str = "dev") -> None:
    index = build(method, stem)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--stem", action="store_true")
    args = parser.parse_args()
    main(args.method, args.stem)