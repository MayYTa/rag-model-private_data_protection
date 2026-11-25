from rag.ingest import clean, split_articles

TEXT = """Artículo 1. Aprobación del decreto
TÍTULO PRELIMINAR
Artículo I. Objeto
texto
Artículo 1. Uno
según el artículo 9.
CAPÍTULO II
“Artículo 2. Modicado
texto
DISPOSICIONES COMPLEMENTARIAS
FINALES
Artículo 3. Fuera
"""


def test_split_keeps_only_the_body():
    rows = split_articles(clean(TEXT), "demo")
    assert [r["articulo"] for r in rows] == ["I", "1", "2"]
    assert "CAPÍTULO" not in rows[1]["texto"]
    assert "Modificado" in rows[2]["texto"]
