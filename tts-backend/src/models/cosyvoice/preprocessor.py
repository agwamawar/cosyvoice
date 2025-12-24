"""Text preprocessing for CosyVoice.

Fully implements:
- Unicode normalization (NFC)
- Abbreviation expansion
- Number to words conversion
- Whitespace cleanup
- Sentence splitting for long texts
"""

import re
import unicodedata
from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger("inference")


# Common abbreviations to expand
ABBREVIATIONS_EN = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "Sr.": "Senior",
    "Jr.": "Junior",
    "St.": "Saint",
    "vs.": "versus",
    "etc.": "et cetera",
    "i.e.": "that is",
    "e.g.": "for example",
    "a.m.": "A M",
    "p.m.": "P M",
    "U.S.": "United States",
    "U.K.": "United Kingdom",
}

# Number words
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
         "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion"]


@dataclass
class PreprocessedText:
    """Result of text preprocessing."""

    original: str
    normalized: str
    phonemes: list[str] | None = None
    tokens: list[int] | None = None
    language: str = "en"


class TextPreprocessor:
    """
    Text preprocessor for CosyVoice TTS.

    Handles:
    - Unicode normalization (NFC)
    - Abbreviation expansion
    - Number to words conversion
    - Whitespace cleanup
    """

    def __init__(self, language: str = "en") -> None:
        self.language = language

    def preprocess(self, text: str, language: str = "en") -> str:
        """
        Preprocess text for TTS synthesis.

        Args:
            text: Raw input text
            language: Language code

        Returns:
            Normalized text string
        """
        lang = language or self.language

        # Step 1: Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # Step 2: Expand abbreviations
        text = self._expand_abbreviations(text, lang)

        # Step 3: Convert numbers to words
        text = self._expand_numbers(text, lang)

        # Step 4: Clean up whitespace
        text = self._clean_whitespace(text)

        # Step 5: Normalize punctuation
        text = self._normalize_punctuation(text)

        return text

    def _expand_abbreviations(self, text: str, language: str) -> str:
        """Expand common abbreviations."""
        if language == "en":
            for abbr, expansion in ABBREVIATIONS_EN.items():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(abbr), re.IGNORECASE)
                text = pattern.sub(expansion, text)
        return text

    def _expand_numbers(self, text: str, language: str) -> str:
        """Convert numbers to words."""
        if language == "en":
            return self._expand_numbers_en(text)
        elif language == "zh":
            return self._expand_numbers_zh(text)
        return text

    def _expand_numbers_en(self, text: str) -> str:
        """Expand numbers to English words."""
        def number_to_words(n: int) -> str:
            """Convert integer to English words."""
            if n == 0:
                return "zero"
            if n < 0:
                return "negative " + number_to_words(-n)

            words = []

            # Handle billions, millions, thousands
            for i, scale in enumerate(reversed(SCALES)):
                divisor = 1000 ** (len(SCALES) - 1 - i)
                if n >= divisor:
                    chunk = n // divisor
                    n %= divisor
                    if chunk > 0:
                        words.append(self._three_digit_to_words(chunk))
                        if scale:
                            words.append(scale)

            return " ".join(words) if words else "zero"

        def replace_number(match: re.Match) -> str:
            num_str = match.group(0)
            try:
                # Handle decimals
                if "." in num_str:
                    parts = num_str.split(".")
                    integer_part = number_to_words(int(parts[0]))
                    decimal_part = " ".join(
                        number_to_words(int(d)) if d != "0" else "zero" 
                        for d in parts[1]
                    )
                    return f"{integer_part} point {decimal_part}"
                else:
                    return number_to_words(int(num_str))
            except (ValueError, OverflowError):
                return num_str

        # Match integers and decimals
        text = re.sub(r"\b\d+(?:\.\d+)?\b", replace_number, text)
        return text

    def _three_digit_to_words(self, n: int) -> str:
        """Convert a three-digit number to words."""
        if n == 0:
            return ""

        words = []

        hundreds = n // 100
        if hundreds > 0:
            words.append(ONES[hundreds])
            words.append("hundred")

        remainder = n % 100
        if remainder > 0:
            if remainder < 10:
                words.append(ONES[remainder])
            elif remainder < 20:
                words.append(TEENS[remainder - 10])
            else:
                tens_digit = remainder // 10
                ones_digit = remainder % 10
                if ones_digit > 0:
                    words.append(f"{TENS[tens_digit]}-{ONES[ones_digit]}")
                else:
                    words.append(TENS[tens_digit])

        return " ".join(words)

    def _expand_numbers_zh(self, text: str) -> str:
        """Expand numbers to Chinese words."""
        chinese_digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

        def replace_digit(match: re.Match) -> str:
            return "".join(chinese_digits[int(d)] for d in match.group(0))

        # Simple digit-by-digit replacement
        text = re.sub(r"\d+", replace_digit, text)
        return text

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        # Normalize dashes
        text = text.replace("—", "-")
        text = text.replace("–", "-")

        # Normalize ellipsis
        text = text.replace("…", "...")

        return text

    def split_long_text(self, text: str, max_length: int = 500) -> list[str]:
        """
        Split long text into chunks at sentence boundaries.

        Args:
            text: Input text
            max_length: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]

        # Split at sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max_length
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If single sentence is too long, split at commas or spaces
                if len(sentence) > max_length:
                    sub_chunks = self._split_sentence(sentence, max_length)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}".strip()

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_sentence(self, sentence: str, max_length: int) -> list[str]:
        """Split a single long sentence."""
        # Try to split at commas first
        parts = re.split(r"(?<=,)\s*", sentence)

        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) + 1 > max_length:
                if current:
                    chunks.append(current.strip())
                if len(part) > max_length:
                    # Last resort: split at spaces
                    words = part.split()
                    for word in words:
                        if len(current) + len(word) + 1 > max_length:
                            if current:
                                chunks.append(current.strip())
                            current = word
                        else:
                            current = f"{current} {word}".strip()
                else:
                    current = part
            else:
                current = f"{current} {part}".strip()

        if current:
            chunks.append(current.strip())

        return chunks


class ChineseTextPreprocessor(TextPreprocessor):
    """Specialized preprocessor for Chinese text."""

    def __init__(self) -> None:
        super().__init__(language="zh")

    def preprocess(self, text: str, language: str = "zh") -> str:
        """Preprocess Chinese text."""
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Convert full-width to half-width punctuation
        text = self._convert_punctuation(text)

        # Expand numbers
        text = self._expand_numbers_zh(text)

        # Clean whitespace
        text = self._clean_whitespace(text)

        return text

    def _convert_punctuation(self, text: str) -> str:
        """Convert full-width punctuation to half-width."""
        conversions = {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "：": ":",
            "；": ";",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "「": '"',
            "」": '"',
        }
        for fw, hw in conversions.items():
            text = text.replace(fw, hw)
        return text


def get_preprocessor(language: str = "en") -> TextPreprocessor:
    """Get appropriate preprocessor for language."""
    if language == "zh":
        return ChineseTextPreprocessor()
    return TextPreprocessor(language=language)
