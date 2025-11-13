# file: dump_scenes.py
from pathlib import Path

# импортируй те же функции, что использует make_features.py
from test import read_script, split_scenes  # должен совпадать с make_features.py

def scene_heading_and_preview(scene, head_n=180):
    # Подстройка под разные представления сцены
    if isinstance(scene, dict):
        head = scene.get("heading") or scene.get("title") or ""
        body = scene.get("text") or scene.get("body") or scene.get("raw") or ""
        text = f"{head} | {body}"
    else:
        text = str(scene)
    text = " ".join(text.split())
    return text[:head_n]

def main(script_path: str, out_path: str = "scenes_dump.txt"):
    text = read_script(script_path)
    scenes = split_scenes(text)
    print(f"Всего сцен: {len(scenes)}")
    lines = []
    for i, sc in enumerate(scenes):
        lines.append(f"{i:04d}: {scene_heading_and_preview(sc)}")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Сцены записаны в {out_path}")

if __name__ == "__main__":
    # Пример: python dump_scenes.py annotatedscript.docx
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dump_scenes.py <script.docx|.txt> [out.txt]")
        sys.exit(1)
    script = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "scenes_dump.txt"
    main(script, out)
