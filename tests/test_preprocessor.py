import pytest
from src.data.preprocessor import TextCleanerPTBR


class TestTextCleanerPTBR:

    def test_clean_empty_and_non_string_inputs(self):
        assert TextCleanerPTBR.clean("") == ""
        assert TextCleanerPTBR.clean("   ") == ""
        assert TextCleanerPTBR.clean(None) == ""
        assert TextCleanerPTBR.clean(123) == ""

    def test_clean_lowercasing(self):
        text = "PRODUTO EXCELENTE E ENTREGA RAPIDA"
        expected = "produto excelente e entrega rapida"
        assert TextCleanerPTBR.clean(text) == expected

    def test_clean_html_tags_and_urls_and_emails(self):
        text = "<p>Comprei no site https://olist.com e mandei email para sac@olist.com.br</p>"
        result = TextCleanerPTBR.clean(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "http" not in result
        assert "sac@" not in result
        assert "comprei no site" in result

    def test_clean_repeated_characters(self):
        text = "ótiiiiiimo produto ameeeeei"
        result = TextCleanerPTBR.clean(text)
        assert "ótiimo" in result
        assert "ameei" in result

    def test_clean_contractions_expansion(self):
        text = "vc comprou tbm pq n gostou ?"
        result = TextCleanerPTBR.clean(text)
        assert "você" in result
        assert "também" in result
        assert "por que" in result
        assert "não" in result

    def test_clean_sentiment_punctuation_and_accents(self):
        text = "Produto péssimo! Não recomendo. Chegou quebrado?"
        result = TextCleanerPTBR.clean(text)
        assert "péssimo!" in result
        assert "não recomendo." in result
        assert "quebrado?" in result

    def test_clean_collapse_extra_spaces(self):
        text = "  muito    bom   recomendo   "
        expected = "muito bom recomendo"
        assert TextCleanerPTBR.clean(text) == expected