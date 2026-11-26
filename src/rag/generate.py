"""Responde con citas usando los artículos recuperados y un modelo local en Ollama.

Uso: python -m rag.generate "¿En cuánto tiempo deben darme acceso a mis datos?" [--trace]
"""
import argparse
import json
import urllib.request

from rag.dense import Dense
from rag.retrieve import load_corpus

URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
ABSTAIN = "No encuentro la respuesta en la normativa disponible."

SYSTEM = """Respondes preguntas sobre protección de datos personales en el Perú.
Usa SOLO las fuentes numeradas que recibes; no uses tu memoria ni normas de otros países.
Si las fuentes no contienen la respuesta, pon "puede_responder": false.
Que una fuente trate un tema parecido no basta: debe contestar lo que se pregunta.
En "fuentes" pon los números de las fuentes que sustentan la respuesta.
Responde en español, breve y claro."""

SCHEMA = {
    "type": "object",
    "properties": {
        "puede_responder": {"type": "boolean"},
        "respuesta": {"type": "string"},
        "fuentes": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["puede_responder", "respuesta", "fuentes"],
}


def build_messages(question: str, rows: list[dict]) -> list[dict]:
    sources = "\n\n".join(f"[{n}] {r['norma']}, artículo {r['articulo']}\n{r['texto']}"
                          for n, r in enumerate(rows, start=1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Fuentes:\n{sources}\n\nPregunta: {question}"}]


def call_ollama(messages: list[dict]) -> str:
    body = {"model": MODEL, "messages": messages, "stream": False, "think": False,
            "format": SCHEMA, "options": {"temperature": 0, "num_ctx": 8192, "seed": 42}}
    request = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["message"]["content"]


def parse(raw: str, rows: list[dict]) -> dict:
    """Valida la salida: las citas solo pueden ser fuentes que sí se entregaron."""
    try:
        data = json.loads(raw)
        numbers = dict.fromkeys(data["fuentes"])
        cited = [rows[n - 1] for n in numbers if 1 <= n <= len(rows)]
        ok = bool(data["puede_responder"]) and bool(cited)
        text = data["respuesta"] if ok else ABSTAIN
    except (json.JSONDecodeError, KeyError, TypeError):
        ok, text, cited = False, ABSTAIN, []
    return {"puede_responder": ok, "respuesta": text,
            "citas": [(r["norma"], r["articulo"]) for r in cited] if ok else []}


class Rag:
    def __init__(self, searcher, llm=call_ollama, trace: bool = False):
        self.search, self.llm = searcher.search, llm
        if trace:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("rag-datos-personales")
            self.search = mlflow.trace(self.search, name="recuperar", span_type="RETRIEVER")
            self.llm = mlflow.trace(self.llm, name="generar", span_type="LLM")
            self.answer = mlflow.trace(self.answer, name="rag")

    def answer(self, question: str, k: int = 5) -> dict:
        rows = self.search(question, k=k)
        return parse(self.llm(build_messages(question, rows)), rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    result = Rag(Dense(load_corpus()), trace=args.trace).answer(args.question)
    print(result["respuesta"])
    for norma, article in result["citas"]:
        print(f"  - {norma}, artículo {article}")


if __name__ == "__main__":
    main()
