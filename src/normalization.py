"""
ML Challenge: Business Entity Resolution
Module: Normalization Pipeline for Business Names and Business Addresses

Handles:
- Case normalization
- Unicode accents, diacritics & Indic script transliterations
- Legal suffix removal
- Common corporate and address abbreviations
- Symbols, ampersands and punctuation
- Dotted acronyms
- State abbreviation expansion for address normalization
- Ordinals and number formatting
- French address nuances
- DBA / trade-name preservation
- Batch processing across large datasets

Important:
- Country is NOT used as a blocking or matching filter.
- Country is used only to interpret address abbreviations correctly.
- Original raw fields are preserved.
"""

import re
import text_unidecode
import pandas as pd
from typing import Optional, Dict, Any

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
# Pre-compiled regular expressions
# ---------------------------------------------------------------------------

RE_COLLAPSE_ACRONYMS = re.compile(
    r'(?<=\b[a-zA-Z])\.(?=\s|[a-zA-Z]|$|[,;:])'
)

RE_URL = re.compile(
    r'https?://\S+|www\.\S+',
    re.IGNORECASE
)

RE_DOMAIN = re.compile(
    r'\b([a-z0-9_-]+)\.(?:com|in|org|net|co|io|gov|edu|fr)\b',
    re.IGNORECASE
)

RE_ORDINALS = re.compile(
    r'\b(\d+)(?:st|nd|rd|th)\b',
    re.IGNORECASE
)

RE_LEADING_ZEROS = re.compile(
    r'\b0+(\d+)\b'
)

RE_PUNCT = re.compile(
    r'[^a-z0-9\s]'
)

RE_SPACES = re.compile(
    r'\s+'
)

# DBA / trade-name markers.
# We capture the part after the marker but do NOT discard the original name.
RE_DBA = re.compile(
    r'\b(?:'
    r'd[\.\/]?b[\.\/]?a\.?'
    r'|t[\.\/]?a\.?'
    r'|f[\.\/]?k[\.\/]?a\.?'
    r'|a[\.\/]?k[\.\/]?a\.?'
    r'|trading as'
    r'|doing business as'
    r')\s*(.+)',
    re.IGNORECASE
)


# Combined state lookup.
# This is ONLY used for address normalization.
# It is NOT used for candidate blocking.
ALL_STATES = {**INDIA_STATES, **US_STATES}


# ---------------------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------------------

def clean_base_text(text: Any) -> str:
    """Basic validation and null-value scrubbing."""

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if text.lower() in (
        'null',
        '<null>',
        '(null)',
        'none',
        'n/a',
        'nan',
        ''
    ):
        return ""

    return text


# ---------------------------------------------------------------------------
# Business-name normalization helpers
# ---------------------------------------------------------------------------

def extract_dba_name(text: str) -> str:
    """
    Extract the trade / DBA portion of a business name.

    Example:
        "ABC Foods Pvt Ltd DBA Krishna Foods"
        -> "Krishna Foods"

    The extracted name is preserved separately rather than replacing
    the original business name.
    """

    match = RE_DBA.search(text)

    if match:
        return match.group(1).strip()

    return ""


def strip_dba_marker(text: str) -> str:
    """
    Remove the DBA/trade-name portion from the main business name.

    Example:
        "ABC Foods Pvt Ltd DBA Krishna Foods"
        -> "ABC Foods Pvt Ltd"

    This allows us to preserve both representations.
    """

    match = RE_DBA.search(text)

    if match:
        return text[:match.start()].strip()

    return text


def normalize_name(text: Any) -> str:
    """
    Normalize the main business name.

    Important:
    DBA / trade names are NOT blindly substituted for the original
    business name. The main normalized name is retained separately
    from the trade name by normalize_record().
    """

    text = clean_base_text(text)

    if not text:
        return ""

    # Remove URL/domain noise.
    text = RE_URL.sub('', text)
    text = RE_DOMAIN.sub(r'\1', text)

    # Preserve only the main business-name portion here.
    # The DBA/trade portion is extracted separately.
    text = strip_dba_marker(text)

    # Remove Indic legal suffixes before transliteration.
    for pat in INDIC_LEGAL_SUFFIXES:
        text = re.sub(
            pat,
            ' ',
            text,
            flags=re.IGNORECASE
        )

    # Transliterate non-Latin scripts and remove accents.
    text = text_unidecode.unidecode(text)

    # Collapse dotted acronyms.
    # Example:
    #   M.G. -> MG
    #   L.L.C. -> LLC
    text = RE_COLLAPSE_ACRONYMS.sub('', text)

    # Lowercase.
    text = text.lower()

    # Symbol normalization.
    text = text.replace('&', ' and ')
    text = text.replace('+', ' and ')
    text = text.replace('@', ' at ')

    # Remove punctuation.
    text = RE_PUNCT.sub(' ', text)

    # Indic phonetic replacements.
    for pat, rep in INDIC_PHONETIC_REPLACEMENTS:
        text = re.sub(pat, rep, text)

    # ---------------------------------------------------------------
    # IMPORTANT CHANGE:
    #
    # Do NOT remove all 7-12 digit numbers.
    #
    # Business names can legitimately contain numbers, so removing
    # them blindly can destroy useful matching information.
    # ---------------------------------------------------------------

    words = text.split()

    if not words:
        return ""

    # ---------------------------------------------------------------
    # Strip legal suffixes from beginning/end iteratively.
    # ---------------------------------------------------------------

    changed = True

    while changed:
        changed = False

        if not words:
            break

        # Check suffix at the end.
        for suffix in LEGAL_SUFFIXES:

            s_words = suffix.split()
            n = len(s_words)

            if len(words) > n and words[-n:] == s_words:
                words = words[:-n]
                changed = True
                break

        # Check suffix at the beginning.
        if not changed:

            for suffix in LEGAL_SUFFIXES:

                s_words = suffix.split()
                n = len(s_words)

                if len(words) > n and words[:n] == s_words:
                    words = words[n:]
                    changed = True
                    break

    # Expand corporate abbreviations.
    expanded_words = [
        NAME_ABBR.get(word, word)
        for word in words
    ]

    text = ' '.join(expanded_words)

    return RE_SPACES.sub(' ', text).strip()


def normalize_trade_name(text: Any) -> str:
    """
    Normalize the DBA / trade-name portion of a business name.

    Example:

        "ABC Foods Pvt Ltd DBA Krishna Foods"

    becomes:

        "krishna foods"
    """

    text = clean_base_text(text)

    if not text:
        return ""

    # Extract DBA/trade portion.
    text = extract_dba_name(text)

    if not text:
        return ""

    # Reuse the same normalization logic.
    return normalize_name(text)


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

def normalize_address(
    text: Any,
    country: Optional[str] = None
) -> str:
    """
    Normalize a business address.

    Country is used ONLY to interpret state abbreviations correctly.

    It is NOT used to:
    - filter records
    - block candidates
    - reject matches
    """

    text = clean_base_text(text)

    if not text:
        return ""

    # ---------------------------------------------------------------
    # 1. Map Indic-script states to English.
    # ---------------------------------------------------------------

    for ind_state, eng_state in INDIC_STATES.items():

        if ind_state in text:
            text = text.replace(
                ind_state,
                f" {eng_state} "
            )

    # ---------------------------------------------------------------
    # 2. Transliteration and accent removal.
    # ---------------------------------------------------------------

    text = text_unidecode.unidecode(text)

    # ---------------------------------------------------------------
    # 3. Collapse dotted acronyms.
    # ---------------------------------------------------------------

    text = RE_COLLAPSE_ACRONYMS.sub('', text)

    # ---------------------------------------------------------------
    # 4. Lowercase.
    # ---------------------------------------------------------------

    text = text.lower()

    # ---------------------------------------------------------------
    # 5. Symbol / address-prefix normalization.
    # ---------------------------------------------------------------

    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('#', ' ')

    text = text.replace(
        'n deg',
        ' number '
    )

    text = text.replace(
        'no.',
        ' no '
    )

    text = text.replace(
        'h.no.',
        ' no '
    )

    text = text.replace(
        'h.no',
        ' no '
    )

    text = text.replace(
        'h no',
        ' no '
    )

    text = text.replace(
        'door no',
        ' no '
    )

    text = text.replace(
        'plot no',
        ' no '
    )

    # ---------------------------------------------------------------
    # 6. Remove embedded null/none/nan tokens.
    # ---------------------------------------------------------------

    text = re.sub(
        r'\b(?:null|none|nan)\b',
        ' ',
        text
    )

    # ---------------------------------------------------------------
    # 7. Normalize ordinals.
    #
    # 45th -> 45
    # 1st  -> 1
    # 2nd  -> 2
    # ---------------------------------------------------------------

    text = RE_ORDINALS.sub(
        r'\1',
        text
    )

    # ---------------------------------------------------------------
    # 8. Remove punctuation.
    # ---------------------------------------------------------------

    text = RE_PUNCT.sub(
        ' ',
        text
    )

    # ---------------------------------------------------------------
    # 9. Remove leading zeros from numbers.
    #
    # 0684 -> 684
    # 022  -> 22
    # ---------------------------------------------------------------

    text = RE_LEADING_ZEROS.sub(
        r'\1',
        text
    )

    # ---------------------------------------------------------------
    # 10. Address abbreviations and state expansion.
    #
    # IMPORTANT:
    # Country only determines which state dictionary is used.
    # It does NOT determine whether records can be matched.
    # ---------------------------------------------------------------

    words = text.split()

    expanded_words = []

    if country == 'US':
        state_map = US_STATES

    elif country == 'India':
        state_map = INDIA_STATES

    else:
        state_map = ALL_STATES

    for word in words:

        if word in ADDRESS_ABBR:
            expanded_words.append(
                ADDRESS_ABBR[word]
            )

        elif word in state_map:
            expanded_words.append(
                state_map[word]
            )

        else:
            expanded_words.append(word)

    text = ' '.join(expanded_words)

    return RE_SPACES.sub(
        ' ',
        text
    ).strip()


# ---------------------------------------------------------------------------
# Record-level normalization
# ---------------------------------------------------------------------------

def normalize_record(
    record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize a single business record.

    Original fields are preserved.

    Additional fields:
        norm_name
        norm_trade_name
        norm_address
    """

    country = record.get('country')

    name = record.get(
        'business_name',
        ''
    )

    address = record.get(
        'business_address',
        ''
    )

    return {
        **record,

        # Main normalized business name.
        'norm_name': normalize_name(name),

        # DBA / trade name kept separately.
        'norm_trade_name': normalize_trade_name(name),

        # Normalized address.
        'norm_address': normalize_address(
            address,
            country=country
        ),
    }


# ---------------------------------------------------------------------------
# DataFrame-level normalization
# ---------------------------------------------------------------------------

def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = 'business_name',
    addr_col: str = 'business_address',
    country_col: Optional[str] = 'country',
    norm_name_col: str = 'norm_name',
    norm_trade_name_col: str = 'norm_trade_name',
    norm_addr_col: str = 'norm_address',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Efficiently normalize a pandas DataFrame.

    Original columns are preserved.

    New columns:
        norm_name
        norm_trade_name
        norm_address
    """

    if not inplace:
        df = df.copy()

    # ---------------------------------------------------------------
    # Normalize business names.
    # ---------------------------------------------------------------

    if name_col in df.columns:

        raw_names = df[name_col].tolist()

        df[norm_name_col] = [
            normalize_name(name)
            for name in raw_names
        ]

        df[norm_trade_name_col] = [
            normalize_trade_name(name)
            for name in raw_names
        ]

    else:

        df[norm_name_col] = ""
        df[norm_trade_name_col] = ""

    # ---------------------------------------------------------------
    # Normalize addresses.
    # ---------------------------------------------------------------

    if addr_col in df.columns:

        raw_addresses = df[addr_col].tolist()

        if (
            country_col
            and country_col in df.columns
        ):

            countries = df[country_col].tolist()

            df[norm_addr_col] = [
                normalize_address(
                    address,
                    country
                )
                for address, country
                in zip(
                    raw_addresses,
                    countries
                )
            ]

        else:

            df[norm_addr_col] = [
                normalize_address(address)
                for address in raw_addresses
            ]

    else:

        df[norm_addr_col] = ""

    return df