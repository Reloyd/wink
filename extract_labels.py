# extract_labels.py
import re, csv, sys
from docx import Document

# Импортируем функции парсинга сцен из твоего скрипта
try:
    from test import split_scenes, parse_header, read_docx_with_cleanup as _read_docx
except Exception:
    # Фоллбек на простое чтение, если функция переименована
    def _read_docx(path: str) -> str:
        doc = Document(path)
        return "\n".join(p.text or "" for p in doc.paragraphs)

# Регулярки для меток
LABEL_RE = re.compile(r'\[\s*(?:Labels|МЕТКИ)\s*:\s*([^\]]+)\]', re.IGNORECASE)
EP_RE    = re.compile(r'\[\s*ep\s*:\s*([^\]]+)\]', re.IGNORECASE)

# Нормализация вариантов ключей и уровней
MAP_KEY = {
    "v":"violence","p":"profanity","s":"sexual","a":"alcohol_drugs","sc":"scary",
    "насилие":"violence","брань":"profanity","секс":"sexual","алкоголь":"alcohol_drugs","страшное":"scary"
}
NORM_SEV = {
    "none":"None","mild":"Mild","moderate":"Moderate","severe":"Severe",
    "нет":"None","лёгкое":"Mild","легкое":"Mild","среднее":"Moderate","жёсткое":"Severe","жесткое":"Severe"
}

def normalize_text(text: str) -> str:
    # Убираем экранирование скобок и хвостовые слэши из .docx
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"[ \t]*\\\\\s*$", "", text, flags=re.MULTILINE)
    # Частые служебные метки от экспорта
    text = text.replace("{.smallcaps}", "")
    return text

def fix_headings(text: str) -> str:
    # убираем хвостовые бэкслэши
    text = re.sub(r"[ \t]*\\\\\s*$", "", text, flags=re.MULTILINE)
    # приводим "ЛОКАЦИЯ.НОЧЬ" к "ЛОКАЦИЯ. - НОЧЬ"
    text = re.sub(r"(\S)\.(\s*)(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО)\b", r"\1. - \3", text, flags=re.IGNORECASE)
    # если время без дефиса: "ЛОКАЦИЯ НОЧЬ", вставим дефис
    text = re.sub(r"(^.*?(ИНТ\.|НАТ\.|INT\.|EXT\.).*?\S)\s+(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО)\b",
                  r"\1 - \3", text, flags=re.IGNORECASE|re.MULTILINE)
    return text


def parse_label_line(line: str):
    labels = {k:"None" for k in ["violence","sexual","profanity","alcohol_drugs","scary"]}
    pairs = [p.strip() for p in line.split(",") if p.strip()]
    for p in pairs:
        if "=" not in p: continue
        k, v = [x.strip() for x in p.split("=", 1)]
        k = MAP_KEY.get(k.lower(), k.lower())
        v = NORM_SEV.get(v.lower(), v)
        if k in labels: labels[k] = v
    return labels

def extract_labels_from_scene(scene_text: str):
    labels = {k:"None" for k in ["violence","sexual","profanity","alcohol_drugs","scary"]}
    m = LABEL_RE.search(scene_text)
    if m:
        labels = parse_label_line(m.group(1))
    # эпизоды (если нужны отдельно, можно сохранить во второй CSV)
    episodes = []
    for em in EP_RE.finditer(scene_text):
        payload = em.group(1)
        # поддержка формата: "v=Severe, sc=Moderate, note=..."
        fields = dict()
        for part in [x.strip() for x in payload.split(",") if x.strip()]:
            if "=" in part:
                k, v = [t.strip() for t in part.split("=", 1)]
                fields[k.lower()] = v
            else:
                fields[part.lower()] = True
        # Перегоняем короткие ключи в канон
        cats = []
        for k_short, k_full in MAP_KEY.items():
            if k_short in ["v","p","s","a","sc"] and k_short in fields:
                cats.append((k_full, NORM_SEV.get(fields[k_short].lower(), fields[k_short])))
        note = fields.get("note","")
        # Если категорий нет, попробуем поле cat
        if not cats and "cat" in fields:
            k = MAP_KEY.get(fields["cat"].lower(), fields["cat"].lower())
            sev = fields.get("sev","Moderate")
            cats.append((k, NORM_SEV.get(sev.lower(), sev)))
        # Сохраняем эпизоды: позиция — начало тега
        for (cat, sev) in cats:
            episodes.append({"cat": cat, "severity": sev})
    return labels, episodes

def read_text(path: str) -> str:
    if path.lower().endswith(".docx"):
        return _read_docx(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main(input_path: str, out_csv: str = "labels.csv"):
    raw = read_text(input_path)
    text = normalize_text(raw)
    text = fix_headings(text)

    # split_scenes может быть в твоём файле; если нет — простой fallback
    try:
        scenes = split_scenes(text)
    except Exception:
        parts = re.split(r'(?=^\s*(?:\d+\s*-\s*\d+\.|СЦЕНА\s*\d*\.|INT\.|EXT\.|ИНТ\.|НАТ\.))',
                         text, flags=re.IGNORECASE | re.MULTILINE)
        scenes = [p.strip() for p in parts if len(p.split()) > 3]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_no","place_type","location","tod",
                    "has_violence","sev_violence",
                    "has_sexual","sev_sexual",
                    "has_profanity","sev_profanity",
                    "has_alcohol_drugs","sev_alcohol_drugs",
                    "has_scary","sev_scary"])
        for s in scenes:
            try:
                meta = parse_header(s)
            except Exception:
                meta = {"scene_no":"","place_type":"","location":"","tod":""}
            labels, _episodes = extract_labels_from_scene(s)
            w.writerow([
                meta.get("scene_no",""), meta.get("place_type",""),
                meta.get("location",""), meta.get("tod",""),
                int(labels["violence"]!="None"), labels["violence"],
                int(labels["sexual"]!="None"), labels["sexual"],
                int(labels["profanity"]!="None"), labels["profanity"],
                int(labels["alcohol_drugs"]!="None"), labels["alcohol_drugs"],
                int(labels["scary"]!="None"), labels["scary"],
            ])
    print(f"✅ OK: извлечено {len(scenes)} сцен → {out_csv}")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "annotated_script.docx"
    main(inp)
