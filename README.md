# RAG sobre protección de datos personales (Perú)

> **English summary:** question answering over Peru's personal data protection law (Law 29733 and
> its regulation, DS 016-2024-JUS) with article-level citations, a local LLM, and an evaluation
> set covering retrieval, citations, answer correctness and abstention.

Responde preguntas sobre la Ley 29733 y su reglamento citando el artículo en que se basa, y se
abstiene cuando no encuentra la respuesta. Todo corre en local. Es un proyecto de aprendizaje:
**no es asesoría legal**.

## Cómo funciona

```
pregunta -> embeddings: 5 artículos más parecidos -> LLM local (Ollama) -> respuesta con citas
```

- **Ingesta:** un registro por artículo (40 de la ley; 144 del reglamento: 9 del Título Preliminar
  y 135 artículos).
- **Recuperación:** `intfloat/multilingual-e5-base`. Los artículos largos se parten en trozos de
  220 palabras con 40 de solape.
- **Generación:** `qwen3.5:9b`, temperatura 0. El modelo cita números de fuente y el código los
  traduce a artículos, así que no puede citar algo que no recibió.
- **Trazas:** MLflow (opcional) registra qué se recuperó y qué respondió el modelo.

## Resultados

El set de evaluación (`eval/eval_set.jsonl`) tiene 40 preguntas divididas en `dev` y `test`.
El método se eligió solo con `dev`.

### Recuperación en dev (19 preguntas con respuesta)
| Método | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|
| BM25 | 0.42 | 0.74 | 0.53 |
| BM25 con stemming | 0.37 | 0.79 | 0.53 |
| **Embeddings** | **0.68** | **0.95** | **0.79** |
| Híbrida (RRF) | 0.58 | 0.74 | 0.65 |

- El *stemming* no dio una mejora clara.
- La híbrida quedó por debajo de los embeddings: con el mismo peso para ambos, BM25 arrastra los
  resultados.

### Recuperación en test (18 preguntas con respuesta)
Embeddings: recall@1 0.56, **recall@5 0.83**, recall@10 0.94, MRR@10 0.67.

### Respuestas en dev
| Medida | Resultado |
|---|---|
| Cita el artículo esperado | 17 de 19 |
| Respuesta correcta (revisión manual) | 16 de 19 |
| Se abstiene cuando debe | 1 de 1 |

La revisión está en `eval/revision_dev.csv`. Las respuestas no se han evaluado en `test`.

### Errores encontrados
| Pregunta | Error | Origen |
|---|---|---|
| `pdp-003` | Dice 20 días; son 8 (art. 69.1). Citó el artículo correcto | Generación |
| `pdp-031` | Dice 48 horas; el aviso es inmediato (art. 36, no recuperado) | Recuperación |
| `pdp-015` | Se abstiene aunque el art. 40 fue recuperado | Generación |

Una cita correcta no garantiza una respuesta correcta.

### Experimento: prompt con reglas adicionales (descartado)
Se añadieron reglas generales (usar el numeral exacto, incluir condiciones, días hábiles).

| En dev (19 preguntas) | Prompt original | Con reglas |
|---|---|---|
| Cita el artículo esperado | 17 | 17 |
| Respuesta correcta (revisión manual) | 16 | 16 |

Corrigió `pdp-003` y `pdp-015`, pero introdujo errores en `pdp-007` y `pdp-025`, y `pdp-031`
siguió mal. Las respuestas salieron más largas, con detalles añadidos que a veces eran erróneos.
Se mantiene el prompt original.

## Limitaciones
- **Muestra pequeña:** cada pregunta pesa unos 5 puntos porcentuales.
- **Preguntas sin verificar:** se redactaron con ayuda de un asistente de IA a partir del texto
  extraído y falta comprobarlas una por una contra el PDF (`verificado_a_mano`).
- **Un artículo esperado por pregunta:** algunas admiten más de uno (por ejemplo, revocar el
  consentimiento: art. 13.7 de la ley y art. 10.1 del reglamento).
- **Abstención casi sin medir:** hay muy pocas preguntas sin respuesta.
- **Corpus:** no incluye las disposiciones complementarias. La ley tiene palabras pegadas por la
  extracción del PDF, y falta contrastar el texto del art. 30 con la fuente oficial.

## Fuentes
| Archivo | URL | Fecha de descarga | Cambios hechos a mano |
|---|---|---|---|
| data/raw/ley_29733.pdf | <https://www.gob.pe/institucion/congreso-de-la-republica/normas-legales/243470-29733> | 25/11/2026 | Se removio una version deprecada de la ley que habia dentro del archivo |
| data/raw/reglamento_ds_016_2024.pdf | <https://www.smv.gob.pe/Uploads/DS016_2024_JUS.pdf> | 25/11/2026 | Ningun cambio fue hecho a mano. |

## Cómo reproducirlo
Requiere Python 3.12 y [Ollama](https://ollama.com).

```bash
pip install -e ".[dense]"
pip install mlflow                 # solo para --trace
ollama pull qwen3.5:9b

python -m rag.ingest               # genera data/processed/
python -m rag.evaluate             # BM25
python -m rag.evaluate --method dense
python -m rag.generate "¿En cuánto tiempo deben darme acceso a mis datos?"
python -m rag.evaluate_answers     # citas y abstención en dev
pytest -q
```

Con una GPU de 8 GB, los embeddings corren en CPU para dejar la GPU al LLM.

Para ver las trazas, añade `--trace` y ejecuta
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Estructura
- `src/rag/`: ingesta, recuperación, generación y evaluación
- `eval/`: preguntas y revisión manual
- `data/raw/`: textos de las normas
- `tests/`: pruebas

## Pendiente
- Verificar el set y ampliar las preguntas sin respuesta
- Reducir los errores de generación sin alargar las respuestas
- Demo pública