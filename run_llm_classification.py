import json
import requests
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "mistral-nemo:latest" # "deepseek-r1:32b"|"llama3.3:latest"|"gemma3:27b"|"mistral-nemo:latest"|"mistral:latest"|"qwen2.5vl:7b"

EVAL_FILE = "evaluation_with_phrase_v4sh.json"  # "evaluation.json"|"evaluation_with_phrase_v2.json"
FEWSHOT_FILE = "fewshot_3shot_v4sh.json" # "fewshot_2shot_with_phrase.json"  # "fewshot_2shot.json"
# DEFINITIONS_FILE = "categories_definitions.json"

OUTPUT_DIR = "predictions_3shot_v4"
OUTPUT_FILE = f"{OUTPUT_DIR}/{MODEL.replace(':', '_')}_3shot_v4sh.json"

TEMPERATURE = 0

# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(entity, phrase, categories, fewshot_examples):

    category_list = "\n".join(
        f"- {category}\n"
        # f"  Definition: {definitions[category]['definition']}"
        for category in categories
    )

    # few-shot examples
    examples = []

    for category, entities in fewshot_examples.items():
        for example_entity in entities:
            examples.append(
                f'Entity: "{example_entity}"\n'
                f'Category: "{category}"'
            )

    examples_text = "\n\n".join(examples)
    
   # prompt
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
# CALL OLLAMA
# ============================================================

def call_ollama(prompt, categories):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string"
                },
                "category": {
                    "type": "string",
                    "enum": categories
                }
            },
            "required": [
                "entity",
                "category"
            ]
        },
        "options": {
            "temperature": TEMPERATURE
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    print("Input tokens:", data.get("prompt_eval_count"))
    print("Output tokens:", data.get("eval_count"))
    print("Prompt duration:", data.get("prompt_eval_duration"))

    return data["message"]["content"]


# ============================================================
# PARSE JSON RESPONSE
# ============================================================

def parse_prediction(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        result = json.loads(text)

        return {
            "entity": result.get("entity"),
            "category": result.get("category"),
            "phrase": result.get("phrase")
        }

    except json.JSONDecodeError:
        print("JSON parsing error:")
        print(text)
        return None


# ============================================================
# SAVE
# ============================================================

def save_predictions(predictions):

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            predictions,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    evaluation = load_json(EVAL_FILE)
    fewshot_examples = load_json(FEWSHOT_FILE)
    # definitions = load_json(DEFINITIONS_FILE)

    categories = list(evaluation.keys())

    safe = {c: c.replace('"', '«', 1).replace('"', '»', 1) for c in categories}
    unsafe = {v: k for k, v in safe.items()}
    safe_categories = list(safe.values())
    safe_fewshot = {safe.get(k, k): v for k, v in fewshot_examples.items()}

    predictions = []

    total = sum(
        len(entities)
        for entities in evaluation.values()
    )

    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Number of categories: {len(categories)}")
    print(f"Evaluation examples: {total}")
    print("=" * 60)

    current = 0

    for gold_category, examples in evaluation.items():

        for example in examples:

            current += 1

            # entity = example["entity"]
            # phrase = example.get("phrase") or ""

            entity = example
            phrase = ""

            print(
                f"\n[{current}/{total}] "
                f"Entity: {entity}"
            )
            print(f"Context: {phrase}")

            prompt = build_prompt(
                entity=entity,
                phrase=phrase,
                categories=safe_categories, # "categories"
                # definitions=definitions,
                fewshot_examples=safe_fewshot # "fewshot_examples"
            )

            try:
                raw_output = call_ollama(prompt, safe_categories)

                parsed = parse_prediction(raw_output)

                if parsed is not None:
                    prediction = parsed["category"]

                    if prediction not in categories:

                        print(
                            f"Invalid category: "
                            f"{prediction}"
                        )
                        prediction = None

            except Exception as e:

                print(f"ERROR: {e}")

                prediction = None

            result = {
                "entity": entity,
                "phrase": phrase,
                "gold": gold_category,
                "prediction": prediction,
                "raw_output": raw_output
            }

            

            predictions.append(result)

            print(f"Gold       : {gold_category}")
            print(f"Prediction : {prediction}")

            # 每跑完一条就保存
            # 即使中途断开，也不会丢掉之前结果
            save_predictions(predictions)

    print("\n" + "=" * 60)
    print("Finished")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()