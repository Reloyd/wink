# annotate_scenes.py
import json, csv, os, re
from test import read_script, split_scenes, parse_header

CHOICES = ["None","Mild","Moderate","Severe"]

def ask_bool(prompt):
    x = input(prompt + " [y/N]: ").strip().lower()
    return x in ["y","yes","д","да"]

def ask_sev(cat):
    x = input(f"Серьёзность {cat} {CHOICES}: ").strip()
    return x if x in CHOICES else "None"

def collect_episodes(scene_text):
    eps = []
    while True:
        add = input("Добавить эпизод? [y/N]: ").strip().lower()
        if add not in ["y","yes","д","да"]:
            break
        text = input("Цитата эпизода: ").strip()
        # найдём первое вхождение как offset; можно вручную указать
        m = re.search(re.escape(text[:12]), scene_text) if text else None
        off = m.start() if m else int(input("offset (число): ").strip() or "0")
        cat = input("Категория [violence/sexual/profanity/alcohol_drugs/scary]: ").strip()
        eps.append({"cat":cat, "offset":off, "text":text})
    return eps

def annotate_file(path, out_csv="labels.csv"):
    text = read_script(path)
    scenes = split_scenes(text)

    exists = os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["scene_no","place_type","location","tod",
                        "has_violence","sev_violence",
                        "has_sexual","sev_sexual",
                        "has_profanity","sev_profanity",
                        "has_alcohol_drugs","sev_alcohol_drugs",
                        "has_scary","sev_scary",
                        "episodes_json","notes"])
        for i, s in enumerate(scenes, 1):
            meta = parse_header(s)
            print("\n"+"="*80)
            print(f"Сцена {i}: {meta}")
            print("-"*80)
            print(s[:1200])  # первые 1200 символов, при нужде увеличить
            print("-"*80)

            hv = ask_bool("Есть насилие?")
            sv = ask_sev("violence") if hv else "None"
            hs = ask_bool("Есть сексуальный контент?")
            ss = ask_sev("sexual") if hs else "None"
            hp = ask_bool("Есть обсценная брань?")
            sp = ask_sev("profanity") if hp else "None"
            ha = ask_bool("Есть алкоголь/наркотики?")
            sa = ask_sev("alcohol_drugs") if ha else "None"
            hc = ask_bool("Есть пугающее?")
            sc = ask_sev("scary") if hc else "None"

            eps = collect_episodes(s)
            notes = input("Примечание: ").strip()

            w.writerow([
                meta.get("scene_no",""), meta.get("place_type",""),
                meta.get("location",""), meta.get("tod",""),
                int(hv), sv, int(hs), ss, int(hp), sp, int(ha), sa, int(hc), sc,
                json.dumps(eps, ensure_ascii=False), notes
            ])
            f.flush()
            cont = input("Дальше? [Enter=да, q=выход]: ").strip().lower()
            if cont == "q":
                break

if __name__ == "__main__":
    path = input("Путь к сценарию: ").strip()
    annotate_file(path)
