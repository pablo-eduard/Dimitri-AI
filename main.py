"""
main.py — Ponto de entrada e orquestrador principal do Dimitri AI.

Controla o loop de eventos, despachando cada rotina no seu intervalo:
  - Notificações (menções/replies) — loop principal
  - Aprendizado via feed home       — a cada INTERVALO_FEED segundos
  - Engajamento proativo            — a cada INTERVALO_BUSCA_TERMOS segundos
  - Atualização de bio              — a cada INTERVALO_BIO_UPDATE segundos
"""
import os
import time
from datetime import datetime

from config import (
    DATA_DIR, LOG_DIR,
    INTERVALO_NOTIFICACOES,
    INTERVALO_FEED,
    INTERVALO_BUSCA_TERMOS,
    INTERVALO_BIO_UPDATE,
    TERMOS_INTERESSE,
    MAX_ENGAJAMENTOS_POR_CICLO,
)
from bluesky_client import BlueskyClient
from admin import GerenciadorAdmin
from web_search import pesquisar_web, deve_pesquisar
from utils.sanitizer import sanitizar_entrada
from utils.logger import logger

# ════════════════════════════════════════════════════════════════
# SELEÇÃO DO BRAIN (Qwen via API ou Markov+TF-IDF local)
# ════════════════════════════════════════════════════════════════
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "").strip()

if HF_API_KEY:
    try:
        from brain_qwen import CerebroConversacional
        TIPO_BRAIN = "Qwen 0.5b (via Hugging Face API)"
    except ImportError:
        from brain import CerebroConversacional
        TIPO_BRAIN = "Markov + TF-IDF (local)"
else:
    from brain import CerebroConversacional
    TIPO_BRAIN = "Markov + TF-IDF (local)"


# ════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════
def garantir_diretorios() -> None:
    """Cria os diretórios de dados e logs se não existirem."""
    for d in (DATA_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# ROTINAS
# ════════════════════════════════════════════════════════════════
def processar_notificacoes(
    cliente: BlueskyClient,
    cerebro: CerebroConversacional,
    admin: GerenciadorAdmin,
) -> None:
    """
    Verifica notificações não lidas e responde a menções/replies.

    Pipeline por notificação:
      1. Sanitiza o texto de entrada
      2. Verifica comandos de admin
      3. Faz pesquisa web se necessário (RAG)
      4. Processa no cérebro e responde
      5. Auto-follow do interagente
    """
    notificacoes = cliente.obter_notificacoes()
    if not notificacoes:
        return

    for notif in notificacoes:
        texto_bruto = getattr(notif.record, "text", "")
        autor = notif.author.handle

        # ── Sanitização ───────────────────────────────────────────
        texto = sanitizar_entrada(texto_bruto)
        if not texto:
            logger.debug(f"[NOTIF] Texto descartado de @{autor}: '{texto_bruto[:40]}'")
            continue

        logger.info(f"[NOTIF] Menção de @{autor}: '{texto[:70]}'")

        # ── Comando de admin ──────────────────────────────────────
        resposta_admin = admin.processar(texto, autor)
        if resposta_admin:
            cliente.responder_post(resposta_admin, notif, autor)
            continue

        # ── RAG: pesquisa web opcional ────────────────────────────
        contexto_externo = None
        if deve_pesquisar(texto):
            contexto_externo = pesquisar_web(texto)
            if contexto_externo:
                logger.debug(f"[RAG] Contexto obtido ({len(contexto_externo)} chars).")

        # ── Identifica thread para memória de curto prazo ─────────
        thread_uri = None
        if hasattr(notif.record, "reply") and notif.record.reply:
            thread_uri = notif.record.reply.root.uri

        # ── Geração de resposta ───────────────────────────────────
        resposta = cerebro.processar_interacao(
            texto,
            thread_uri=thread_uri,
            contexto_externo=contexto_externo,
        )

        if resposta:
            enviado = cliente.responder_post(resposta, notif, autor)
            # Auto-follow apenas se conseguiu responder
            if enviado and hasattr(notif.author, "did"):
                cliente.seguir_usuario(notif.author.did, autor)

    cliente.marcar_notificacoes_lidas()


def rotina_aprendizado_feed(
    cliente: BlueskyClient,
    cerebro: CerebroConversacional,
) -> None:
    """
    Lê o feed home e injeta textos higienizados no cérebro.
    Executa periodicamente para garantir aprendizado contínuo e orgânico.
    """
    logger.info("[FEED] Iniciando rotina de aprendizado via feed home...")
    textos = cliente.obter_feed_home(limite=30)
    injetados = sum(1 for t in textos if cerebro.injetar_no_cerebro(t))
    logger.info(f"[FEED] {injetados}/{len(textos)} textos injetados no cérebro.")


def rotina_engajamento_proativo(
    cliente: BlueskyClient,
    cerebro: CerebroConversacional,
) -> None:
    """
    Busca posts por termos de interesse e comenta proativamente.

    Limita a MAX_ENGAJAMENTOS_POR_CICLO comentários por rodada para
    evitar comportamento de spam.
    """
    logger.info("[PROATIVO] Iniciando busca de termos de interesse...")
    total_comentarios = 0

    for termo in TERMOS_INTERESSE:
        if total_comentarios >= MAX_ENGAJAMENTOS_POR_CICLO:
            break

        posts = cliente.buscar_posts_por_termo(termo, limite=5)
        for post in posts:
            if total_comentarios >= MAX_ENGAJAMENTOS_POR_CICLO:
                break

            texto_sanitizado = sanitizar_entrada(post["texto"])
            if not texto_sanitizado:
                continue

            resposta = cerebro.processar_interacao(texto_sanitizado)
            if not resposta:
                continue

            sucesso = cliente.comentar_post(resposta, post["uri"], post["cid"])
            if sucesso:
                total_comentarios += 1
                logger.info(
                    f"[PROATIVO] Comentou no post de @{post['autor']} "
                    f"(termo: '{termo}')"
                )
                time.sleep(5)  # Delay entre comentários proativos

    logger.info(f"[PROATIVO] {total_comentarios} comentário(s) realizado(s) neste ciclo.")


def rotina_atualizar_bio(
    cliente: BlueskyClient,
    cerebro: CerebroConversacional,
) -> None:
    """
    Atualiza a bio do perfil do Dimitri AI com estatísticas em tempo real.
    """
    s = cerebro.obter_status()
    bio = (
        f"🤖 Dimitri AI | Bot em aprendizado contínuo\n"
        f"🧠 Memórias acumuladas: {s['total_memorias']}\n"
        f"⚡ Última atualização: {s['timestamp']}\n"
        f"💡 Powered by Markov + TF-IDF | Processamento 100% local"
    )
    cliente.atualizar_bio(bio)


# ════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ════════════════════════════════════════════════════════════════
def executar_bot() -> None:
    """
    Ponto de entrada principal. Inicializa todos os componentes e
    entra no loop de eventos com controle de tempo para cada rotina.
    """
    garantir_diretorios()

    logger.info("=" * 60)
    logger.info("  DIMITRI AI — Inicializando sistema...")
    logger.info(f"  Motor de IA: {TIPO_BRAIN}")
    logger.info("=" * 60)

    # Inicializa os componentes principais
    cliente = BlueskyClient()
    cerebro = CerebroConversacional()
    admin = GerenciadorAdmin(cerebro)

    # Tenta conectar ao Bluesky (falha fatal se não conseguir)
    try:
        cliente.conectar()
    except Exception as exc:
        logger.critical(f"Falha crítica no login. Encerrando. Detalhe: {exc}")
        return

    # Marca notificações passadas como lidas (evita responder histórico)
    cliente.marcar_notificacoes_lidas()
    logger.info("✅ Dimitri AI está online. Monitorando notificações...")

    # Controladores de tempo para rotinas secundárias
    ultimo_feed = 0.0
    ultimo_engajamento = 0.0
    ultima_bio = 0.0

    while True:
        agora = time.time()

        # ── Rotina principal: notificações ────────────────────────
        try:
            processar_notificacoes(cliente, cerebro, admin)
        except Exception as exc:
            logger.error(f"Erro no loop de notificações: {exc}", exc_info=True)

        # ── Rotina de aprendizado via feed ────────────────────────
        if agora - ultimo_feed >= INTERVALO_FEED:
            try:
                rotina_aprendizado_feed(cliente, cerebro)
            except Exception as exc:
                logger.error(f"Erro na rotina de feed: {exc}", exc_info=True)
            finally:
                ultimo_feed = agora

        # ── Rotina de engajamento proativo ────────────────────────
        if agora - ultimo_engajamento >= INTERVALO_BUSCA_TERMOS:
            try:
                rotina_engajamento_proativo(cliente, cerebro)
            except Exception as exc:
                logger.error(f"Erro no engajamento proativo: {exc}", exc_info=True)
            finally:
                ultimo_engajamento = agora

        # ── Rotina de atualização de bio ──────────────────────────
        if agora - ultima_bio >= INTERVALO_BIO_UPDATE:
            try:
                rotina_atualizar_bio(cliente, cerebro)
            except Exception as exc:
                logger.error(f"Erro na atualização de bio: {exc}", exc_info=True)
            finally:
                ultima_bio = agora

        time.sleep(INTERVALO_NOTIFICACOES)


if __name__ == "__main__":
    executar_bot()
