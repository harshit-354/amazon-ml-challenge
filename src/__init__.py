"""
Business Entity Resolution System
"""

from src.normalization import (
    normalize_name,
    normalize_address,
    normalize_record,
    normalize_dataframe,
)

__all__ = [
    "normalize_name",
    "normalize_address",
    "normalize_record",
    "normalize_dataframe",
]
