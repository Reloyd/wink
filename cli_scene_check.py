# cli_scene_check.py — проверка одной сцены: категории + финальный возраст

import json
import sys
import re
import numpy as np

from test import (
    analyze_scene,           # возвращает per_class по категориям
    age_from_scene,          # возрастной рейтинг из per_class (без оверрайдов)
    legal_overrides          # юридические оверрайды (18+ кейсы)
)

def load_input_text():
    if len(sys.argv) >= 2:
        path_or_text = " ".join(sys.argv[1:])
        # Если это путь к файлу — прочтем, иначе считаем аргументацию как текст
        try:
            with open(path_or_text, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return path_or_text
    # STDIN
    data = sys.stdin.read()
    if data.strip():
        return data
    print("Usage:\n  python cli_scene_check.py \"Текст сцены...\"\n  python cli_scene_check.py path/to/scene.txt\n  cat scene.txt | python cli_scene_check.py")
    sys.exit(1)

def main():
    scene_text = load_input_text()

    # 1) Пер‑категорийный анализ
    per_class = analyze_scene(scene_text)

    # 2) Предварительный возраст по ML/правилам
    base_age = age_from_scene(per_class)

    # 3) Юридические оверрайды (приоритетно)
    override = legal_overrides(scene_text)
    final_age = base_age
    if override:
        order = ["0+","6+","12+","16+","18+"]
        if order.index(override["min_age"]) > order.index(base_age):
            final_age = override["min_age"]

    # 4) Компактный вывод
    out = {
        "base_age": base_age,
        "final_age": final_age,
        "override": override or None,
        "per_class": {
            k: {
                "rule_score": float(v["rule_score"]),
                "model_proba": float(v["model_proba"]),
                "episode_max": float(v["episode_max"]),
                "final_proba": float(v["final_proba"]),
                "severity": v["severity"],
                "episodes_count": int(len(v.get("episodes", [])))
            } for k, v in per_class.items()
        }
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
