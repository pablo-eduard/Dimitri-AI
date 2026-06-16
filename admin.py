"""
admin.py — Processador de comandos administrativos do Dimitri AI.

O dono do bot pode enviar comandos via Bluesky. O bot verifica o
handle do autor antes de processar qualquer comando privilegiado.
"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from config import ADMIN_HANDLE, COMANDOS_ADMIN
from utils.logger import logger

if TYPE_CHECKING:
    from brain import CerebroConversacional


class GerenciadorAdmin:
    """
    Verifica e executa comandos enviados pelo administrador do bot.

    Apenas mensagens de ADMIN_HANDLE são aceitas.
    """

    def __init__(self, cerebro: "CerebroConversacional") -> None:
        self._cerebro = cerebro

    def eh_admin(self, handle: str) -> bool:
        """Verifica se o handle é do administrador autorizado."""
        if not ADMIN_HANDLE:
            return False  # Admin não configurado — nenhum comando aceito
        return handle.lower() == ADMIN_HANDLE.lower()

    def processar(self, texto: str, handle: str) -> Optional[str]:
        """
        Verifica se o texto contém um comando de admin e o executa.

        Args:
            texto:  Texto da mensagem recebida.
            handle: Handle do autor da mensagem.

        Returns:
            Resposta do comando executado, ou None se não for um comando válido.
        """
        if not self.eh_admin(handle):
            return None

        texto_lower = texto.lower()
        for cmd, acao in COMANDOS_ADMIN.items():
            if cmd in texto_lower:
                logger.info(f"[ADMIN] Comando '{cmd}' recebido de @{handle}")
                return self._executar(acao)

        return None

    def _executar(self, acao: str) -> str:
        """Despacha a ação para o handler correspondente."""
        handlers = {
            "clear_memory": self._limpar_memoria,
            "reload_model": self._recarregar_modelo,
            "show_status":  self._mostrar_status,
        }
        handler = handlers.get(acao)
        if handler:
            return handler()
        return f"Ação desconhecida: '{acao}'."

    def _limpar_memoria(self) -> str:
        self._cerebro.limpar_memoria()
        return "✅ Memória de longo prazo apagada com sucesso!"

    def _recarregar_modelo(self) -> str:
        self._cerebro.recarregar_modelo()
        return "✅ Modelo Markov e cache TF-IDF recarregados!"

    def _mostrar_status(self) -> str:
        s = self._cerebro.obter_status()
        return (
            f"📊 Status do Dimitri AI\n"
            f"Memórias: {s['total_memorias']}\n"
            f"Markov: {'✅' if s['modelo_markov_ativo'] else '❌'}\n"
            f"TF-IDF: {'✅' if s['tfidf_disponivel'] else '❌'}\n"
            f"Hora: {s['timestamp']}"
        )
