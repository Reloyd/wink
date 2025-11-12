# debug_split.py — диагностика пропущенных сцен
import re
from docx import Document
from normalize import normalize_headings

def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text or "" for p in doc.paragraphs)

# Текущий паттерн из вашего кода
MULTI_PATTERN_SPLIT = re.compile(
    r'(?=^\s*'
    r'(?:'
    r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
    r'(?:\d{1,2}-[ЕE]\.?)?\s*'
    r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?|И/Н|I/E)?\s*'
    r'[^\n]{3,140}?'
    r'\s*[-–—:]\s*'
    r'(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|EVENING|MORNING)\b'
    r'|'
    r'\d+\.\s*(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s+[^\n]{3,120}\s*[-–—]\s*(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT)\b'
    r'|'
    r'(?:\d+\s*-\s*\d+)?\.?\s*[иінінат]+\.?\s+[^\n]{3,120}\s*[-–—]\s*(?:день|ночь|вечер|утро)\b'
    r'|'
    r'СЦЕНА\s+\d*'
    r')'
    r')',
    re.IGNORECASE | re.MULTILINE
)

def analyze_splits(path):
    text = read_docx(path)
    text = normalize_headings(text)
    
    # Найти все заголовки вручную (не через split)
    header_pattern = re.compile(
        r'^\s*(?:'
        r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
        r'(?:\d{1,2}-[ЕE]\.?)?\s*'
        r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'
        r'[^\n]{3,140}?'
        r'\s*[-–—]\s*'
        r'(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT)\b'
        r')',
        re.IGNORECASE | re.MULTILINE
    )
    
    all_headers = list(header_pattern.finditer(text))
    print(f"📊 Найдено заголовков (через finditer): {len(all_headers)}\n")
    
    # Разбить через split
    parts = re.split(MULTI_PATTERN_SPLIT, text)
    scenes = [p.strip() for p in parts if len(p.split()) >= 5]
    print(f"📊 Получено сцен (через split): {len(scenes)}\n")
    
    # Вывести первые 10 заголовков
    print("=" * 70)
    print("ПЕРВЫЕ 10 ЗАГОЛОВКОВ (через finditer):")
    print("=" * 70)
    for i, m in enumerate(all_headers[:10], 1):
        header = m.group(0).strip()
        print(f"{i}. {header}")
    
    print("\n" + "=" * 70)
    print("ПЕРВЫЕ 10 СЦЕН (через split):")
    print("=" * 70)
    for i, s in enumerate(scenes[:10], 1):
        first_line = s.splitlines()[0][:120] if s.splitlines() else s[:120]
        print(f"{i}. {first_line}")
    
    # Проверка пропусков
    if len(all_headers) > len(scenes):
        print(f"\n⚠️ ПРОБЛЕМА: Пропущено {len(all_headers) - len(scenes)} заголовков!")
        print("Возможные причины:")
        print("1. Lookahead не срабатывает на некоторых форматах")
        print("2. Фильтр len(p.split()) >= 5 отбрасывает короткие сцены")
        print("3. Между сценами есть служебный текст (титры, номера серий)")

if __name__ == "__main__":
    path = input("Путь к сценарию: ").strip()
    analyze_splits(path)
