"""
Classification des sources (28 catégories) sur tout le corpus, via Ollama (serveur SSH).

Même prompt, mêmes exemples et même format de sortie que connect_llm.py (évaluation v4).

Entrée  : spans_qeXVI_clean.json      (un objet JSON par ligne, un span par ligne)
Sorties : predictions_cache.json      (un objet par couple unique entité + phrase ; sert à la reprise)
          spans_qeXVI_classified.json (même format et même ordre que l'entrée + champ « category »)

Relancer le script reprend là où il s'est arrêté.

Sur le serveur :
    OLLAMA_NUM_PARALLEL=8 ollama serve
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11435/api/chat"
MODEL = "mistral-nemo:latest"

SPANS_FILE = "spans_qeXVI_clean.json"
FEWSHOT_FILE = "fewshot_3shot_v4.json"
CACHE_FILE = "predictions_cache.json"
OUTPUT_FILE = "spans_qeXVI_classified.json"

TEMPERATURE = 0
N_WORKERS = 8      # = OLLAMA_NUM_PARALLEL sur le serveur
NUM_CTX = 8192     # le prompt fait ~3 300-3 500 tokens


# ============================================================
# BUILD PROMPT  (identique à connect_llm.py)
# ============================================================

fewshot_examples = json.load(open(FEWSHOT_FILE, encoding="utf-8"))
categories = list(fewshot_examples.keys())

category_list = "\n".join(f"- {category}\n" for category in categories)

examples_text = "\n\n".join(
    f'Entity: "{example_entity}"\n'
    f'Category: "{category}"'
    for category, entities in fewshot_examples.items()
    for example_entity in entities
)


def build_prompt(entity, phrase):
    prompt = f"""
You are performing CLOSED-SET entity classification.

Classify the entity into exactly ONE category from the list below.

CATEGORIES:
{category_list}

FEW-SHOT EXAMPLES:
{examples_text}

ENTITY TO CLASSIFY:
"{entity}"

OPTIONAL CONTEXT:
"{phrase}"

Use the context only if it helps identify what the entity refers to,
for example when the entity is an acronym, abbreviation, shortened name,
or ambiguous expression.

The final category must describe the entity itself.

Do not classify based on the topic, action, or other information in the sentence.

If the entity is already clear from its name, ignore the context.

RULES:
1. Choose exactly one category from the list.
2. Copy the category name exactly as written.
3. Do not invent or rename categories.
4. Output only valid JSON.

{{
  "entity": "{entity}",
  "category": "<exact category name>"
}}
"""
    return prompt.strip()


# ============================================================
# CALL OLLAMA  (même payload que connect_llm.py + num_ctx)
# ============================================================

session = requests.Session()


def classify(entity, phrase):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(entity, phrase)}],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "category": {"type": "string", "enum": categories},
            },
            "required": ["entity", "category"],
        },
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    r = session.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    category = json.loads(data["message"]["content"]).get("category")
    if category not in categories:
        raise ValueError(f"catégorie invalide : {category!r}")
    return {
        "category": category,
        "input_tokens": data.get("prompt_eval_count"),
        "seconds": data.get("total_duration", 0) / 1e9,
    }


# ============================================================
# MAIN
# ============================================================

def key(entity, phrase):
    return f"{entity}\t{phrase}"


def main():
    # données : un couple (entité, phrase) n'est classé qu'une fois ; entités vides ignorées
    spans = [json.loads(l) for l in open(SPANS_FILE, encoding="utf-8") if l.strip()]
    pairs = list(dict.fromkeys(
        ((r.get("entity_clean") or "").strip(), r.get("text") or "")
        for r in spans
        if (r.get("entity_clean") or "").strip()
    ))

    # reprise
    done = {}
    if Path(CACHE_FILE).exists():
        for line in open(CACHE_FILE, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[key(rec["entity"], rec["phrase"])] = rec["category"]

    todo = [(e, p) for e, p in pairs if key(e, p) not in done]

    print("=" * 60)
    print(f"Model: {MODEL} | workers: {N_WORKERS} | num_ctx: {NUM_CTX}")
    print(f"{len(spans)} spans, {len(pairs)} couples uniques, {len(done)} déjà faits, {len(todo)} à faire")
    print("=" * 60)

    lock = threading.Lock()
    first = True
    n_err = 0
    with open(CACHE_FILE, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(classify, e, p): (e, p) for e, p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            e, p = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # sauté ici, refait au prochain lancement
                n_err += 1
                tqdm.write(f"ERROR {e!r}: {exc}")
                continue
            with lock:
                fout.write(json.dumps({"entity": e, "phrase": p, **res}, ensure_ascii=False) + "\n")
                fout.flush()
            done[key(e, p)] = res["category"]
            if first:
                tqdm.write(f"Input tokens (1re requête) : {res['input_tokens']}")
                first = False

    # sortie : même format et même ordre que le fichier d'entrée
    n_missing = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in spans:
            e = (r.get("entity_clean") or "").strip()
            r["category"] = done.get(key(e, r.get("text") or "")) if e else None
            n_missing += r["category"] is None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Saved to: {OUTPUT_FILE} | sans prédiction : {n_missing} | erreurs : {n_err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
