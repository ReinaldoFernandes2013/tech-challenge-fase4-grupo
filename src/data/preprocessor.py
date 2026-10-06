import re
import unicodedata


class TextCleanerPTBR:
    """Higienizador e normalizador de texto voltado para PT-BR."""

    CONTRACTIONS = {
        "vc": "você",
        "tb": "também",
        "tbm": "também",
        "pq": "por que",
        "obg": "obrigado",
        "blz": "beleza",
        "n": "não",
        "naum": "não",
        "q": "que",
    }

    @classmethod
    def clean(cls, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Normalização Unicode para forma composta estável (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Minúsculas
        text = text.lower()

        # 3. Remoção de URLs, tags HTML e e-mails
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # 4. Normalização de caracteres repetidos (ex.: "ótiiimo" -> "ótimo")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 5. Expansão de contrações informais
        tokens = text.split()
        tokens = [cls.CONTRACTIONS.get(t, t) for t in tokens]
        text = " ".join(tokens)

        # 6. Filtragem mantendo acentuação PT-BR e pontuações úteis para sentimento
        text = re.sub(r"[^a-záàãâéêíóôõúüç\s.,!?]", "", text)

        # 7. Colapso de múltiplos espaços
        text = re.sub(r"\s+", " ", text).strip()
        return text