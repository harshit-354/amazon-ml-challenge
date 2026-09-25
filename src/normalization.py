"""
ML Challenge: Business Entity Resolution
Module: Normalization Pipeline for Business Names and Business Addresses

Handles:
- Case normalization (lowercase)
- Unicode accents, diacritics & Indic script transliterations (Devanagari, Tamil, etc.)
- Legal suffixes removal and standardization (Pvt Ltd, LLC, Inc, Corp, SARL, SAS, etc.)
- Common corporate and address abbreviations (Rd -> road, St -> street, Ave -> avenue, etc.)
- Symbols, ampersands, punctuation, special noise characters (& -> and, + -> and)
- Dotted acronyms (M.G. -> mg, C.I.T. -> cit, U.S.A. -> usa)
- State abbreviations expansion (country-aware: NY -> new york, MH -> maharashtra, etc.)
- Ordinals and number formatting (1st -> 1, 45th -> 45, 0684 -> 684)
- French address nuances (BD -> boulevard, R. -> rue, N° -> number)
- Fast vectorized / batch processing across large datasets
"""

import re
import unicodedata
import text_unidecode
import pandas as pd
from typing import Optional, List, Dict, Any, Union

from src.constants import (
    INDIC_STATES,
    INDIC_LEGAL_SUFFIXES,
    US_STATES,
    INDIA_STATES,
    ADDRESS_ABBR,
    LEGAL_SUFFIXES,
    NAME_ABBR,
    INDIC_PHONETIC_REPLACEMENTS,
)

# ---------------------------------------------------------------------------
# Pre-compiled regular expressions for speed
# ---------------------------------------------------------------------------
RE_COLLAPSE_ACRONYMS = re.compile(r'(?<=\b[a-zA-Z])\.(?=\s|[a-zA-Z]|$|[,;:])')
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_DOMAIN = re.compile(r'\b([a-z0-9_-]+)\.(?:com|in|org|net|co|io|gov|edu|fr)\b', re.IGNORECASE)
RE_ORDINALS = re.compile(r'\b(\d+)(?:st|nd|rd|th)\b', re.IGNORECASE)
RE_LEADING_ZEROS = re.compile(r'\b0+(\d+)\b')
RE_PHONE = re.compile(r'\b(?:ph|tel|phone|fax)?\.?\s*(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,6}\b', re.IGNORECASE)
RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_SPACES = re.compile(r'\s+')
RE_DBA = re.compile(
    r'\b(?:d[\.\/]?b[\.\/]?a\.?|t[\.\/]?a\.?|f[\.\/]?k[\.\/]?a\.?|a[\.\/]?k[\.\/]?a\.?|trading as|doing business as)\s*(.+)',
    re.IGNORECASE
)
RE_TRIPLE_CHARS = re.compile(r'([a-z])\1{2,}')

# Combined state lookup
ALL_STATES = {**INDIA_STATES, **US_STATES}


def clean_base_text(text: Any) -> str:
    """Basic validation and null-value scrubbing."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if text.lower() in ('null', '<null>', '(null)', 'none', 'n/a', 'nan', ''):
        return ""
    return text


def normalize_name(text: Any) -> str:
    """
    Normalize a business name string.
    
    Transforms:
    1. Null / empty handling
    2. URL and domain extension stripping
    3. DBA / Trade name handling (extracting true business name)
    4. Indic legal suffix handling prior to transliteration
    5. Transliteration of non-Latin scripts (Devanagari, Tamil, etc.) and accents removal
    6. Dotted acronym collapsing (e.g. M.G. -> MG)
    7. Lowercasing
    8. Symbol replacement (& -> and, + -> and, @ -> at)
    9. Punctuation and bracket removal
    10. Legal suffix stripping from start and end (Pvt Ltd, LLC, Inc, Corp, SARL, SAS, etc.)
    11. Common corporate abbreviation expansion (mfg -> manufacturing, tech -> technology)
    12. Whitespace normalization
    """
    text = clean_base_text(text)
    if not text:
        return ""

    # 1. URL / Domain stripping (e.g., maurewilliamscolombier.com -> maurewilliamscolombier)
    text = RE_URL.sub('', text)
    text = RE_DOMAIN.sub(r'\1', text)

    # 2. Extract true entity name from DBA / trade name patterns if present
    dba_match = RE_DBA.search(text)
    if dba_match:
        text = dba_match.group(1)

    # 3. Strip Indic legal suffixes in native scripts
    for pat in INDIC_LEGAL_SUFFIXES:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)

    # 4. Transliterate non-Latin scripts & strip accents
    text = text_unidecode.unidecode(text)

    # 5. Collapse dotted acronyms (e.g., M.G. -> MG, L.L.C. -> LLC)
    text = RE_COLLAPSE_ACRONYMS.sub('', text)

    # 6. Lowercase
    text = text.lower()

    # 7. Convert symbols
    text = text.replace('&', ' and ')
    text = text.replace('+', ' and ')
    text = text.replace('@', ' at ')

    # 8. Clean punctuation, brackets and special chars
    text = RE_PUNCT.sub(' ', text)

    # 9. Clean triple duplicate letters from transliteration (e.g., innnvesttmenntts -> investtmenntts)
    text = RE_TRIPLE_CHARS.sub(r'\1', text)

    # 10. Indic phonetic transliteration replacements
    for pat, rep in INDIC_PHONETIC_REPLACEMENTS:
        text = re.sub(pat, rep, text)

    # 11. Remove phone numbers / long standalone digit strings
    text = re.sub(r'\b\d{7,12}\b', '', text)

    # 12. Strip legal suffixes from start and end iteratively
    words = text.split()
    if not words:
        return ""

    changed = True
    while changed:
        changed = False
        if not words:
            break
        # Check from end
        for suffix in LEGAL_SUFFIXES:
            s_words = suffix.split()
            n = len(s_words)
            if len(words) > n and words[-n:] == s_words:
                words = words[:-n]
                changed = True
                break
        # Check from start (e.g., 'LLC Crystal Staffing Solutions')
        if not changed:
            for suffix in LEGAL_SUFFIXES:
                s_words = suffix.split()
                n = len(s_words)
                if len(words) > n and words[:n] == s_words:
                    words = words[n:]
                    changed = True
                    break

    # 13. Expand standard corporate abbreviations
    expanded_words = [NAME_ABBR.get(w, w) for w in words]
    text = ' '.join(expanded_words)

    return RE_SPACES.sub(' ', text).strip()


def normalize_address(text: Any, country: Optional[str] = None) -> str:
    """
    Normalize a business address string.
    
    Transforms:
    1. Null / empty handling
    2. Indic script state mapping to canonical English (Maharashtra, Karnataka, etc.)
    3. Transliteration of accents and non-Latin scripts (French accents, Indic scripts)
    4. Dotted acronym collapsing (M.G. -> MG, N.C. -> NC)
    5. Lowercasing
    6. French & international number markers (N°, № -> number)
    7. Symbol replacement (& -> and, @ -> at, # -> space)
    8. Unit prefix normalization (h.no, h no, door no -> no)
    9. Embedded null / none removal
    10. Ordinals standardization (45th / 45nd -> 45, 1st -> 1, 2nd -> 2)
    11. Punctuation removal
    12. Leading zeros removal in numbers (0684 -> 684, 022 -> 22)
    13. Address abbreviations expansion (rd -> road, st/saint -> street, blvd/bd -> boulevard, etc.)
    14. State abbreviation expansion (country-aware or global)
    15. Whitespace normalization
    """
    text = clean_base_text(text)
    if not text:
        return ""

    # 1. Map Indic script states to English
    for ind_state, eng_state in INDIC_STATES.items():
        if ind_state in text:
            text = text.replace(ind_state, f" {eng_state} ")

    # 2. Transliterate accents and non-Latin scripts
    text = text_unidecode.unidecode(text)

    # 3. Collapse dotted acronyms (e.g., M.G. -> MG, N.C. -> NC)
    text = RE_COLLAPSE_ACRONYMS.sub('', text)

    # 4. Lowercase
    text = text.lower()

    # 5. Convert symbols and prefixes
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('#', ' ')
    text = text.replace('n deg', ' number ')
    text = text.replace('no.', ' no ')
    text = text.replace('h.no.', ' no ')
    text = text.replace('h.no', ' no ')
    text = text.replace('h no', ' no ')
    text = text.replace('door no', ' no ')
    text = text.replace('plot no', ' no ')

    # 6. Remove embedded null / none / nan keywords
    text = re.sub(r'\b(?:null|none|nan)\b', ' ', text)

    # 7. Standardize ordinals (e.g., 45th / 45nd -> 45, 1st -> 1, 2nd -> 2)
    text = RE_ORDINALS.sub(r'\1', text)

    # 8. Remove punctuation
    text = RE_PUNCT.sub(' ', text)

    # 9. Strip leading zeros in numbers (e.g., 0684 -> 684, 022 -> 22)
    text = RE_LEADING_ZEROS.sub(r'\1', text)

    # 10. Address abbreviations and state expansion
    words = text.split()
    expanded_words = []

    # Choose state mapping
    if country == 'US':
        state_map = US_STATES
    elif country == 'India':
        state_map = INDIA_STATES
    else:
        state_map = ALL_STATES

    for w in words:
        if w in ADDRESS_ABBR:
            expanded_words.append(ADDRESS_ABBR[w])
        elif w in state_map:
            expanded_words.append(state_map[w])
        else:
            expanded_words.append(w)

    text = ' '.join(expanded_words)
    return RE_SPACES.sub(' ', text).strip()


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single record dictionary containing business_name and business_address."""
    country = record.get('country')
    name = record.get('business_name', '')
    addr = record.get('business_address', '')
    
    return {
        **record,
        'norm_name': normalize_name(name),
        'norm_address': normalize_address(addr, country=country),
    }


def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = 'business_name',
    addr_col: str = 'business_address',
    country_col: Optional[str] = 'country',
    norm_name_col: str = 'norm_name',
    norm_addr_col: str = 'norm_address',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Efficiently normalize a pandas DataFrame of business entities.
    Uses fast Python list comprehensions for high throughput.
    """
    if not inplace:
        df = df.copy()

    # Fast list comprehension normalization
    raw_names = df[name_col].tolist() if name_col in df.columns else []
    df[norm_name_col] = [normalize_name(n) for n in raw_names]

    raw_addrs = df[addr_col].tolist() if addr_col in df.columns else []
    if country_col and country_col in df.columns:
        countries = df[country_col].tolist()
        df[norm_addr_col] = [normalize_address(a, c) for a, c in zip(raw_addrs, countries)]
    else:
        df[norm_addr_col] = [normalize_address(a) for a in raw_addrs]

    return df
