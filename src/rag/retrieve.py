"""BM25 sobre artículos: un documento por artículo."""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import snowballstemmer

STOP = set(
    ["a", "al", "ante", "como", "con", "cual", "cuales", "cuando", "de", "del", "desde", "donde", "e", "el", "ella", "en", "entre", "es", "esta", "este", "hay", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "no", "o", "para", "por", "puede", "que", "se", "si", "sin", "sobre", "son", "su", "sus", "tiene", "un", "una", "uno", "unos", "y", "ya"]
)


STEMMER = snowballstemmer.stemmer("spanish")


def tokenize(text: str, stem: bool = False) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    words = [w for w in words if strip_accents(w) not in STOP]
    if stem:  # el stemmer necesita las tildes, por eso se quitan después
        words = STEMMER.stemWords(words)
    return [strip_accents(w) for w in words]


def strip_accents(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    return plain.decode()


def load_corpus(folder: str = "data/processed") -> list[dict]:
    files = (Path(folder) / n for n in ("ley.jsonl", "reglamento.jsonl"))
    return [json.loads(x) for f in files for x in f.read_text(encoding="utf-8").splitlines() if x]


class BM25:
    def __init__(self, rows: list[dict], k1: float = 1.5, b: float = 0.75,
                 stem: bool = False):
        self.rows, self.k1, self.b, self.stem = rows, k1, b, stem
        self.docs = [Counter(tokenize(r["texto"], stem)) for r in rows]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.mean = sum(self.lengths) / len(self.lengths)
        df = Counter(t for d in self.docs for t in d)
        n = len(rows)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, terms: list[str], i: int) -> float:
        doc = self.docs[i]
        norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.mean)
        return sum(self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                   for t in terms if t in doc)

    def search(self, query: str, k: int = 5) -> list[dict]:
        terms = tokenize(query, self.stem)
        order = sorted(range(len(self.rows)), key=lambda i: -self.score(terms, i))
        return [self.rows[i] for i in order[:k]]
