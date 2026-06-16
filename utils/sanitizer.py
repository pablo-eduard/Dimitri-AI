"""
utils/sanitizer.py — Sanitização de entradas, saídas e conteúdo do cérebro.

Garante que o Dimitri AI não aprenda, reproduza ou processe conteúdo
inadequado, links maliciosos ou padrões de spam.
"""
import re
from typing import Optional

from config import (
    PALAVRAS_PROIBIDAS, PADROES_SPAM,
    TEXTO_MIN_CHARS, TEXTO_MAX_CHARS, LIMITE_CARACTERES_POST,
)
from utils.logger import logger


def _contem_spam(texto: str) -> bool:
    """Retorna True se o texto corresponder a algum padrão de spam."""
    for padrao in PADROES_SPAM:
        if re.search(padrao, texto, re.IGNORECASE):
            return True
    return False


def _contem_palavras_proibidas(texto: str) -> bool:
    """Retorna True se o texto contiver alguma palavra proibida."""
    texto_lower = texto.lower()
    return any(p in texto_lower for p in PALAVRAS_PROIBIDAS)


def sanitizar_entrada(texto: str) -> Optional[str]:
    """
    Valida e limpa o texto de entrada recebido de usuários externos.

    Pipeline:
    1. Remove menções (@handle) para não poluir o processamento NLP.
    2. Remove espaços excessivos.
    3. Rejeita texto vazio, muito curto, muito longo, com spam ou proibido.

    Args:
        texto: Texto bruto recebido da API do Bluesky.

    Returns:
        Texto limpo e validado, ou None se inválido.
    """
    if not texto or not isinstance(texto, str):
        return None

    # Remove menções
    texto = re.sub(r"@[\w.]+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    if len(texto) < TEXTO_MIN_CHARS:
        return None

    if len(texto) > TEXTO_MAX_CHARS:
        logger.debug(f"[SANITIZER] Texto truncado (era {len(texto)} chars).")
        texto = texto[:TEXTO_MAX_CHARS]

    if _contem_spam(texto):
        logger.warning(f"[SANITIZER] Spam detectado: '{texto[:50]}'")
        return None

    if _contem_palavras_proibidas(texto):
        logger.warning(f"[SANITIZER] Conteúdo proibido detectado: '{texto[:50]}'")
        return None

    return texto


def sanitizar_saida(texto: str) -> str:
    """
    Garante que a resposta do bot seja segura e dentro do limite de caracteres.

    Args:
        texto: Resposta gerada pelo cérebro antes do envio.

    Returns:
        Texto seguro, dentro do limite, pronto para postar.
    """
    if not texto or not texto.strip():
        return "Hmm, preciso de um momento para processar isso!"

    # Remove links gerados acidentalmente
    texto = re.sub(PADROES_SPAM[0], "[link]", texto)

    # Verifica palavras proibidas na saída
    if _contem_palavras_proibidas(texto):
        logger.warning(f"[SANITIZER] Palavra proibida na saída: '{texto[:50]}'")
        return "Cada conversa me ensina algo novo. Me fala mais!"

    # Respeita o limite de caracteres do Bluesky
    if len(texto) > LIMITE_CARACTERES_POST:
        # Tenta cortar em um ponto final para não quebrar uma frase
        truncado = texto[:LIMITE_CARACTERES_POST - 3]
        ultimo_ponto = truncado.rfind(".")
        if ultimo_ponto > LIMITE_CARACTERES_POST // 2:
            return truncado[:ultimo_ponto + 1]
        return truncado + "..."

    return texto


def higienizar_para_cerebro(texto: str) -> Optional[str]:
    """
    Prepara texto do feed externo para ser injetado no cerebro_inicial.txt.
    É mais restritivo que sanitizar_entrada, pois afeta o treinamento.

    Args:
        texto: Texto bruto de um post alheio no Bluesky.

    Returns:
        Texto limpo para injeção, ou None se inadequado.
    """
    resultado = sanitizar_entrada(texto)
    if not resultado:
        return None

    # Remove caracteres especiais que atrapalham o tokenizador Markov
    resultado = re.sub(
        r"[^\w\s.,!?;:àáâãéêíóôõúüçÀÁÂÃÉÊÍÓÔÕÚÜÇ–—]",
        " ",
        resultado,
    )
    resultado = re.sub(r"\s+", " ", resultado).strip()

    # Mínimo de 15 chars para valer a pena como dado de treinamento
    if len(resultado) < 15:
        return None

    return resultado
