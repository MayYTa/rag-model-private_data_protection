"""Evalúa citas y abstención. Uso: python -m rag.evaluate_answers [--split dev] [--trace]"""
import argparse
import json
from pathlib import Path

from rag.dense import Dense
from rag.generate import Rag
from rag.retrieve import load_corpus


def cites_gold(result: dict, gold: dict) -> bool:
    article = gold["articulos"][0].split(".")[0]
    return any(a == article and gold["norma"].startswith(n) for n, a in result["citas"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    lines = Path("eval/eval_set.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [q for q in map(json.loads, lines) if q["split"] == args.split]
    rag = Rag(Dense(load_corpus()), trace=args.trace)
    hits, abstained, records = [], [], []
    for q in questions:
        result = rag.answer(q["pregunta"])
        if q["articulos"]:
            ok = cites_gold(result, q)
            hits.append(ok)
        else:
            ok = not result["puede_responder"]
            abstained.append(ok)
        print(f"{q['id']} {'bien' if ok else 'REVISAR'}: {result['respuesta'][:90]}")
        records.append({**q, **result})
    Path(f"data/processed/respuestas_{args.split}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"cita el artículo esperado: {sum(hits)}/{len(hits)}")
    print(f"se abstiene cuando debe: {sum(abstained)}/{len(abstained)}")


if __name__ == "__main__":
    main()
