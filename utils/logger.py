"""
utils/logger.py — Logger centralizado do Dimitri AI.

Usa RotatingFileHandler para não lotar o disco, e StreamHandler
para exibir logs INFO+ no console em tempo real.
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def configurar_logger(
    nome: str = "DimitriAI",
    arquivo_log: str = "logs/dimitri.log",
    nivel_console: int = logging.INFO,
    nivel_arquivo: int = logging.DEBUG,
) -> logging.Logger:
    """
    Cria e configura o logger principal do bot.

    Args:
        nome:          Nome do logger (aparece nos registros).
        arquivo_log:   Caminho para o arquivo de log.
        nivel_console: Nível mínimo para exibição no console.
        nivel_arquivo: Nível mínimo para gravação em arquivo.

    Returns:
        Logger configurado e pronto para uso.
    """
    os.makedirs(os.path.dirname(arquivo_log), exist_ok=True)

    logger = logging.getLogger(nome)

    # Evita duplicação de handlers se o módulo for reimportado
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formato = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Handler de arquivo (rotação a cada 5 MB, 5 backups) ──────────────
    fh = RotatingFileHandler(
        arquivo_log,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(nivel_arquivo)
    fh.setFormatter(formato)

    # ── Handler de console ────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(nivel_console)
    ch.setFormatter(formato)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# Instância global — importe de qualquer módulo com: from utils.logger import logger
logger = configurar_logger()
