# normalize.py — финальная версия с полным покрытием edge cases
import re
import unicodedata


def normalize_headings(text: str) -> str:
    """
    Комплексная нормализация заголовков сцен с поддержкой всех форматов.
    """
    
    # ===== ЭТАП 1: Unicode-нормализация =====
    text = unicodedata.normalize('NFC', text)
    
    # ===== ЭТАП 2: Унификация разделителей =====
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[ \t]*\\\\\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]*\\\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("{.smallcaps}", "").replace("{.underline}", "")
    text = text.replace("\\[", "[").replace("\\]", "]")
    
    # ===== ЭТАП 3: Нормализация типов мест (ИНТ/НАТ) =====
    # 3.1) Поднимаем строчные в начале строки
    text = re.sub(
        r'(?im)^\s*(нат|инт|nat|int|ext)\b\.?',
        lambda m: m.group(1).upper() + '.',
        text
    )
    
    # 3.2) Добавляем точку если отсутствует после ИНТ/НАТ
    text = re.sub(
        r'(?im)\b(ИНТ|НАТ|INT|EXT)(?!\.)\s+',
        r'\1. ',
        text
    )
    
    # 3.3) Убираем дефис/двоеточие ПОСЛЕ типа места: "ИНТ. - " -> "ИНТ. "
    text = re.sub(
        r'(?im)\b(ИНТ\.|НАТ\.|INT\.|EXT\.)\s*[-:]\s+',
        r'\1 ',
        text
    )
    
    # ===== ЭТАП 4: Обработка слитного написания =====
    # 4.1) "ИНТ.ЛОКАЦИЯ.ВРЕМЯ" -> "ИНТ. ЛОКАЦИЯ - ВРЕМЯ"
    # Используем lookahead для проверки времени суток в конце
    text = re.sub(
        r'\b(ИНТ|НАТ|INT|EXT)\.([A-ZА-ЯЁ][^\.\n]{2,80})\.(?=ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|EVENING|MORNING\b)',
        r'\1. \2 - ',
        text,
        flags=re.IGNORECASE
    )

    # ===== ЭТАП 4.7: "ЛОКАЦИЯ. ВРЕМЯ" -> "ЛОКАЦИЯ - ВРЕМЯ" (для формата с номером) =====
    text = re.sub(
        r'(?im)^(\d+\.\s*(?:ИНТ\.|НАТ\.|INT\.|EXT\.)\s+[^\n]{2,120}?)\.\s*'
        r'(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)\b',
        r'\1 - \2',
        text,
        flags=re.MULTILINE
    )

    # ===== ЭТАП 4.8: "НАТ.У ЦИРКА. ДЕНЬ" -> "НАТ. У ЦИРКА - ДЕНЬ" =====
    text = re.sub(
        r'(?im)^(\d+\.\s*(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*[A-ZА-ЯЁ]{3,})\.'
        r'([A-ZА-ЯЁ]{3,})\s*\.\s*(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT)',
        r'\1 \2 - \3',
        text,
        flags=re.MULTILINE
    )
    
    # ===== ЭТАП 5: Нормализация разделителя между локацией и временем =====
    # 5.1) Двоеточие или точка перед временем -> дефис
    text = re.sub(
        r'(?im)([^-:\n]{2,120}?)\s*[:.]\s*-?\s*(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|EVENING|MORNING)\b',
        r'\1 - \2',
        text
    )
    
    # 5.2) Убираем точку в конце времени суток
    text = re.sub(
        r'(?im)\b(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|EVENING|MORNING)\.\s*$',
        r'\1',
        text,
        flags=re.MULTILINE
    )
    
    # ===== ЭТАП 6: Унификация пробелов =====
    # Множественные пробелы -> один пробел
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Пробелы в конце строк
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    
    # ===== ЭТАП 7: Двойные дефисы =====
    text = re.sub(r'\s*--+\s*', ' - ', text)
    
    # Множественные пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def normalize_scene_heading_strict(heading: str) -> str:
    """
    Строгая нормализация отдельного заголовка.
    """
    # ===== ШАГ 0: Unicode-нормализация =====
    heading = unicodedata.normalize('NFC', heading)
    
    # ===== ШАГ 1: Разделители =====
    heading = heading.replace("–", "-").replace("—", "-").replace("−", "-")
    heading = heading.replace("\\", "")
    heading = heading.replace("{.smallcaps}", "").replace("{.underline}", "")
    
    # ===== ШАГ 2: Тип места в верхний регистр + точка =====
    heading = re.sub(
        r'\b(нат|инт|nat|int|ext)\b\.?',
        lambda m: m.group(1).upper() + '.',
        heading,
        flags=re.IGNORECASE
    )
    
    # ===== ШАГ 3: Добавляем точку после типа если отсутствует =====
    heading = re.sub(
        r'\b(ИНТ|НАТ|INT|EXT)(?!\.)\s+',
        r'\1. ',
        heading
    )

    # ШАГ 3.1: Гарантируем один пробел после типа ("ИНТ.МИЛИЦИЯ" -> "ИНТ. МИЛИЦИЯ")
    heading = re.sub(r'\b(ИНТ\.|НАТ\.|INT\.|EXT\.)\s*([^\s\-])', r'\1 \2', heading) 
    
    # ===== ШАГ 4: Убираем дефис/двоеточие после типа =====
    heading = re.sub(
        r'\b(ИНТ\.|НАТ\.|INT\.|EXT\.)\s*[-:]\s*',
        r'\1 ',
        heading
    )
    
    # ===== ШАГ 5: Слитное написание "ЛЕС.ОПУШКА" =====
    # Разделяем слова через точку без пробела
    heading = re.sub(
        r'\b([A-ZА-ЯЁ]{3,})\.([A-ZА-ЯЁ][A-ZА-ЯЁ]{2,})\b',
        r'\1. \2',
        heading
    )

    # ===== ШАГ 5.5: Разделяем точки в локациях БЕЗ времени после (для 12, 13) =====
    heading = re.sub(
        r'\b(ИНТ\.|НАТ\.|INT\.|EXT\.)\s*([A-ZА-ЯЁ]{3,})\.([A-ZА-ЯЁ]{3,})(?!\s*(?:ДЕНЬ|НОЧЬ|[-–—]))',
        r'\1 \2. \3',
        heading
    )
    
    # ===== ШАГ 6: Время в верхний регистр =====
    heading = re.sub(
        r'\b(день|ночь|вечер|утро|day|night|evening|morning|режим|рассвет|закат|сумерки)\b',
        lambda m: m.group(1).upper(),
        heading,
        flags=re.IGNORECASE
    )
    
    # ===== ШАГ 7: Разделитель между локацией и временем =====
    heading = re.sub(
        r'([^-:\n]{2,120}?)\s*:\s*-?\s*(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?',
        r'\1 - \2',
        heading
    )

# ===== ШАГ 8: НЕ трогаем точки в формате "ЛОКАЦИЯ. ВРЕМЯ" =====
# Только добавляем дефис если нет НИКАКОГО разделителя
    heading = re.sub(
        r'([A-ZА-ЯЁ][^\n-:.]{2,120}?)\s{2,}(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ)(?:\s+\d+)?(?!\s*[-.])',
        r'\1 - \2',
        heading
    )

    # ===== ШАГ 8: Если дефиса нет, добавляем =====
    heading = re.sub(
        r'([A-ZА-ЯЁ][^\n-]{2,120}?)\s+(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ|ЗАКАТ|СУМЕРКИ)(?:\s+\d+)?(?!\s*-)',
        r'\1 - \2',
        heading
    )
    
    # ===== ШАГ 9: Убираем точку после времени =====
    heading = re.sub(
        r'\b(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|EVENING|MORNING)(?:\s+\d+)?\.',
        r'\1',
        heading
    )
    
    # ===== ШАГ 10: Множественные пробелы =====
    heading = re.sub(r'\s{2,}', ' ', heading).strip()
    
    # ===== ШАГ 11: Двойные дефисы =====
    heading = re.sub(r'\s*--+\s*', ' - ', heading)
    
    return heading


# ===== ТЕСТИРОВАНИЕ =====
if __name__ == "__main__":
    test_cases = [
        "1-2 инт. ПОЕЗД. КУПЕ - ДЕНЬ.",
        "1-2-А НАТ.: УЛИЦА - НОЧЬ",
        "3. ИНТ - КОМНАТА: день",
        "ИНТ.ЛОКАЦИЯ.ВЕЧЕР",
        "5 ИНТ КУХНЯ — УТРО",
        "НАТ - ДОМ.ДЕНЬ",
        "инт. квартира: ночь.",
        "ИНТ: ОФИС - ДЕНЬ",
    ]
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ normalize_scene_heading_strict()")
    print("=" * 60)
    for case in test_cases:
        normalized = normalize_scene_heading_strict(case)
        print(f"Исходный:      {case}")
        print(f"Нормализован:  {normalized}\n")
