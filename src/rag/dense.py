"""Recuperación por significado (embeddings) e híbrida."""

MODEL = "intfloat/multilingual-e5-base"


def chunks(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    """Parte un texto en trozos de palabras, para no pasar el límite de 512 tokens."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL)  # usa la GPU si está disponible


class Dense:
    def __init__(self, rows: list[dict], model=None):
        self.rows = rows
        self.model = model or load_model()
        self.owner, passages = [], []
        for i, row in enumerate(rows):
            for piece in chunks(row["texto"]):
                self.owner.append(i)
                passages.append(f"passage: {row['norma']}. {piece}")
        self.vectors = self.model.encode(passages, normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[dict]:
        vector = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        best: dict[int, float] = {}
        for i, score in zip(self.owner, self.vectors @ vector):
            best[i] = max(score, best.get(i, -1.0))  # un artículo vale lo que su mejor trozo
        return [self.rows[i] for i in sorted(best, key=lambda i: -best[i])[:k]]


class Hybrid:
    """Reciprocal Rank Fusion: suma 1/(c + posición) de cada buscador."""

    def __init__(self, rows: list[dict], searchers: list, c: int = 60):
        self.rows, self.searchers, self.c = rows, searchers, c

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores: dict[tuple, float] = {}
        found: dict[tuple, dict] = {}
        for searcher in self.searchers:
            for position, row in enumerate(searcher.search(query, k=50), start=1):
                key = (row["norma"], row["articulo"])
                found[key] = row
                scores[key] = scores.get(key, 0.0) + 1 / (self.c + position)
        return [found[key] for key in sorted(scores, key=lambda x: -scores[x])[:k]]
