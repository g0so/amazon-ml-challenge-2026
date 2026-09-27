"""Phase 2 normalization: Unicode NFC, case folding, and whitespace only.

No abbreviation expansion or punctuation/numeric removal. Tokens such as CT,
FL, St, and N have ambiguous meanings. Those belong in later tested features,
not unconditional replacements in this conservative baseline.
"""
import unicodedata

NORMALIZATION_VERSION = 'nfc-casefold-whitespace-v1'


def normalize_text(raw):
    if not isinstance(raw, str):
        raise TypeError('Expected a string; preserve missing TSV fields as empty strings')
    return ' '.join(unicodedata.normalize('NFC', raw).casefold().split())


def normalize_name(raw):
    return raw, normalize_text(raw)


def normalize_address(raw):
    return raw, normalize_text(raw)


def normalize_record(raw_name, raw_address):
    name_raw, name_norm = normalize_name(raw_name)
    address_raw, address_norm = normalize_address(raw_address)
    return {'name_raw': name_raw, 'name_norm': name_norm,
            'address_raw': address_raw, 'address_norm': address_norm}
