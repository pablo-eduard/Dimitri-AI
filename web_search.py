"""
web_search.py — Pesquisa web leve para enriquecer respostas do Dimitri AI (RAG simples).

Usa DuckDuckGo Search (sem necessidade de API key) para buscar
trechos contextuais quando o usuário faz perguntas que demandam
informações atualizadas ou que não estão no corpus de Markov.
"""
import re
from typing import Optional

from utils.logger import logger

try:
    from duckduckgo_search import DDGS
    DDGS_DISPONIVEL = True
    logger.info("DuckDuckGo Search disponível — pesquisa web ativa.")
except ImportError:
    DDGS_DISPONIVEL = False
    logger.warning(
        "duckduckgo_search não instalado. Pesquisa web desabilitada. "
        "Instale com: pip install duckduckgo-search"
    )

# Marcadores linguísticos que indicam intenção de busca de informação
_MARCADORES_BUSCA = (
    "o que é", "o que são", "me explica", "me fala sobre",
    "como funciona", "quem é", "quando foi", "onde fica",
    "qual é", "você sabe", "me conta sobre", "você conhece",
    "notícia", "atualidade", "última", "recente", "hoje",
    "lançamento", "novo", "novidade",
)

# Comprimento máximo do contexto retornado (em caracteres)
_MAX_CONTEXTO_CHARS = 500
_MAX_CHARS_POR_RESULTADO = 220


def deve_pesquisar(texto: str) -> bool:
    """
    Heurística rápida para decidir se vale a pena pesquisar na web.

    Args:
        texto: Texto sanitizado do usuário.

    Returns:
        True se o texto contiver marcadores de busca.
    """
    texto_lower = texto.lower()
    return any(m in texto_lower for m in _MARCADORES_BUSCA)


def pesquisar_web(query: str, max_resultados: int = 3) -> Optional[str]:
    """
    Realiza uma busca no DuckDuckGo e consolida os trechos relevantes.

    Os trechos são higienizados (sem HTML, espaços normalizados) e
    limitados em tamanho para não inflar o contexto além do útil.

    Args:
        query:           Termo de busca (normalmente o texto do usuário).
        max_resultados:  Número máximo de resultados a considerar.

    Returns:
        String com trechos concatenados, ou None em caso de falha/indisponibilidade.
    """
    if not DDGS_DISPONIVEL:
        return None

    try:
        trechos = []
        with DDGS() as ddgs:
            resultados = ddgs.text(
                query,
                region="br-pt",
                safesearch="moderate",
                max_results=max_resultados,
            )
            for r in resultados:
                corpo = r.get("body", "").strip()
                if not corpo or len(corpo) < 30:
                    continue
                # Normaliza espaços e trunca
                trecho = re.sub(r"\s+", " ", corpo[:_MAX_CHARS_POR_RESULTADO]).strip()
                trechos.append(trecho)

        if not trechos:
            logger.debug(f"[RAG] Sem resultados para: '{query[:50]}'")
            return None

        contexto = " | ".join(trechos)
        contexto = contexto[:_MAX_CONTEXTO_CHARS]
        logger.info(f"[RAG] {len(trechos)} trecho(s) obtidos para: '{query[:50]}'")
        return contexto

    except Exception as exc:
        logger.error(f"[RAG] Erro na pesquisa web para '{query[:50]}': {exc}")
        return None
