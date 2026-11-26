import json

from rag.generate import ABSTAIN, parse

ROWS = [{"norma": "Ley 29733", "articulo": "14", "texto": "x"}]


def reply(**changes) -> str:
    return json.dumps({"puede_responder": True, "respuesta": "Sí.", "fuentes": [1]} | changes)


def test_maps_source_numbers_to_articles():
    assert parse(reply(), ROWS)["citas"] == [("Ley 29733", "14")]


def test_abstains_without_valid_citation():
    for raw in (reply(fuentes=[7]), reply(puede_responder=False), "no es json"):
        result = parse(raw, ROWS)
        assert result["respuesta"] == ABSTAIN and result["citas"] == []
