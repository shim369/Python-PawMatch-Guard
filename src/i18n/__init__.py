"""Minimal i18n: English (default) and Japanese.

The language is chosen with the ``PAWMATCH_LANG`` environment variable
(``en`` / ``ja``) or the ``--lang`` CLI flag. Unknown keys fall back to English,
then to the key itself, so a missing translation never crashes the program.
"""

from __future__ import annotations

import os

from .messages import MESSAGES

SUPPORTED = ("en", "ja")
DEFAULT = "en"


def get_lang(lang: str | None = None) -> str:
    """Resolve the active language code."""
    code = (lang or os.environ.get("PAWMATCH_LANG") or DEFAULT).lower()[:2]
    return code if code in SUPPORTED else DEFAULT


def t(key: str, lang: str | None = None, **kwargs: object) -> str:
    """Translate ``key`` and format it with ``kwargs``."""
    code = get_lang(lang)
    text = MESSAGES.get(code, {}).get(key) or MESSAGES[DEFAULT].get(key) or key
    return text.format(**kwargs) if kwargs else text
