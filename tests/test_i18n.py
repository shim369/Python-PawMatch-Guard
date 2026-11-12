import string

from src.i18n import t
from src.i18n.messages import MESSAGES


def _fields(s: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(s) if f}


def test_catalogs_have_same_keys_and_placeholders() -> None:
    en, ja = MESSAGES["en"], MESSAGES["ja"]
    assert en.keys() == ja.keys()
    for k in en:
        assert _fields(en[k]) == _fields(ja[k]), k


def test_translate_and_fallback(monkeypatch) -> None:
    assert t("cat", "en") == "cat"
    assert t("cat", "ja") == "猫"
    assert t("cat", "fr") == "cat"  # unsupported -> English
    assert t("no.such.key", "ja") == "no.such.key"
    monkeypatch.setenv("PAWMATCH_LANG", "ja")
    assert t("dog") == "犬"
