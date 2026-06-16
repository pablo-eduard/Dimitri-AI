"""
bluesky_client.py — Interface completa com a rede social Bluesky via atproto.

Responsabilidades:
  - Login com reconexão automática (exponential backoff)
  - Recebimento e filtragem de notificações
  - Postagem de respostas com rate limiting
  - Leitura de feed home (aprendizado contínuo)
  - Busca proativa por termos
  - Auto-follow de interagentes
  - Atualização dinâmica de bio
"""
import time
import functools
from typing import Any, Callable, Dict, List, Optional

from atproto import Client, models
from atproto_client.exceptions import AtProtocolError

from config import (
    BLUESKY_HANDLE, BLUESKY_APP_PASSWORD,
    BACKOFF_BASE, BACKOFF_MAX, BACKOFF_TENTATIVAS,
    DELAY_ENTRE_POSTS, MAX_POSTS_POR_HORA,
    LIMITE_CARACTERES_POST, validar_credenciais_bluesky,
)
from utils.logger import logger


# ════════════════════════════════════════════════════════════════
# DECORATOR: EXPONENTIAL BACKOFF
# ════════════════════════════════════════════════════════════════
def com_backoff(func: Callable) -> Callable:
    """
    Decorator que adiciona retry com exponential backoff a chamadas de API.

    Captura AtProtocolError e erros de rede genéricos, retentatando com
    espera crescente. Lança a exceção após esgotar as tentativas.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        delay = 1.0
        for tentativa in range(1, BACKOFF_TENTATIVAS + 1):
            try:
                return func(*args, **kwargs)
            except (AtProtocolError, ConnectionError, TimeoutError) as exc:
                if tentativa >= BACKOFF_TENTATIVAS:
                    logger.error(
                        f"[BACKOFF] '{func.__name__}' falhou definitivamente "
                        f"após {BACKOFF_TENTATIVAS} tentativas: {exc}"
                    )
                    raise
                delay = min(BACKOFF_BASE ** tentativa, BACKOFF_MAX)
                logger.warning(
                    f"[BACKOFF] '{func.__name__}' tentativa {tentativa}/{BACKOFF_TENTATIVAS} "
                    f"falhou. Aguardando {delay:.1f}s... | Erro: {exc}"
                )
                time.sleep(delay)
        return None
    return wrapper


# ════════════════════════════════════════════════════════════════
# RATE LIMITER
# ════════════════════════════════════════════════════════════════
class RateLimiter:
    """
    Controla a frequência de postagens para respeitar os limites da API.

    Mantém um histórico deslizante de timestamps de postagem.
    """

    def __init__(self, max_por_hora: int, delay_minimo: float) -> None:
        self._max_por_hora = max_por_hora
        self._delay_minimo = delay_minimo
        self._historico: List[float] = []
        self._ultimo_post: float = 0.0

    def pode_postar(self) -> bool:
        """Verifica se ainda há quota disponível na última hora."""
        agora = time.time()
        # Desliza a janela — remove registros com mais de 1 hora
        self._historico = [t for t in self._historico if agora - t < 3600]
        restante = self._max_por_hora - len(self._historico)
        if restante <= 0:
            logger.warning(
                f"[RATE LIMIT] Quota esgotada ({self._max_por_hora} posts/hora)."
            )
        return restante > 0

    def aguardar_delay_minimo(self) -> None:
        """Bloqueia até que o delay mínimo entre posts seja respeitado."""
        elapsed = time.time() - self._ultimo_post
        if elapsed < self._delay_minimo:
            time.sleep(self._delay_minimo - elapsed)

    def registrar_post(self) -> None:
        """Registra um novo post no histórico."""
        agora = time.time()
        self._historico.append(agora)
        self._ultimo_post = agora


# ════════════════════════════════════════════════════════════════
# CLIENTE BLUESKY
# ════════════════════════════════════════════════════════════════
class BlueskyClient:
    """
    Wrapper de alto nível para a API do Bluesky via atproto.

    Encapsula autenticação, rate limiting, backoff e todas as
    operações necessárias para o Dimitri AI funcionar na rede.
    """

    def __init__(self) -> None:
        self._client = Client()
        self._rate = RateLimiter(MAX_POSTS_POR_HORA, DELAY_ENTRE_POSTS)
        self._conectado = False

    # ────────────────────────────────────────────────────────────
    # CONEXÃO
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def conectar(self) -> bool:
        """Realiza o login e marca o cliente como conectado."""
        validar_credenciais_bluesky()
        logger.debug(f"[LOGIN] Tentando login com handle={BLUESKY_HANDLE!r}")
        self._client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)
        self._conectado = True
        logger.info(f"[LOGIN] Autenticado como @{BLUESKY_HANDLE}")
        return True

    # ────────────────────────────────────────────────────────────
    # NOTIFICAÇÕES
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def obter_notificacoes(self) -> List:
        """
        Obtém e filtra notificações não lidas relevantes.

        Retorna apenas menções e replies que não sejam do próprio bot.
        """
        dados = self._client.app.bsky.notification.list_notifications()
        filtradas = [
            n for n in dados.notifications
            if not n.is_read
            and n.reason in ("mention", "reply")
            and n.author.handle != BLUESKY_HANDLE
        ]
        if filtradas:
            logger.info(f"[NOTIF] {len(filtradas)} notificação(ões) não lida(s).")
        return filtradas

    def marcar_notificacoes_lidas(self) -> None:
        """Atualiza o timestamp de 'visto' para limpar a fila de notificações."""
        try:
            self._client.app.bsky.notification.update_seen(
                {"seen_at": self._client.get_current_time_iso()}
            )
            logger.debug("[NOTIF] Notificações marcadas como lidas.")
        except Exception as exc:
            logger.warning(f"[NOTIF] Erro ao marcar como lidas: {exc}")

    # ────────────────────────────────────────────────────────────
    # POSTAGEM DE RESPOSTAS
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def responder_post(
        self,
        texto: str,
        notificacao: Any,
        autor_handle: str,
    ) -> bool:
        """
        Envia uma resposta em um fio de conversa existente.

        Respeita rate limiting e constrói corretamente as referências
        de parent e root para manter a thread íntegra.

        Args:
            texto:          Texto da resposta (já sanitizado e limitado).
            notificacao:    Objeto de notificação da API.
            autor_handle:   Handle do autor para logging.

        Returns:
            True se o post foi enviado com sucesso.
        """
        if not self._rate.pode_postar():
            return False

        self._rate.aguardar_delay_minimo()

        # Trunca com segurança
        if len(texto) > LIMITE_CARACTERES_POST:
            texto = texto[:LIMITE_CARACTERES_POST - 3] + "..."

        # Monta as referências do fio
        parent_ref = models.ComAtprotoRepoStrongRef.Main(
            cid=notificacao.cid, uri=notificacao.uri
        )
        root_ref = parent_ref

        if hasattr(notificacao.record, "reply") and notificacao.record.reply:
            root_ref = models.ComAtprotoRepoStrongRef.Main(
                cid=notificacao.record.reply.root.cid,
                uri=notificacao.record.reply.root.uri,
            )

        reply_ref = models.AppBskyFeedPost.ReplyRef(
            parent=parent_ref, root=root_ref
        )
        self._client.send_post(text=texto, reply_to=reply_ref)
        self._rate.registrar_post()

        logger.info(f"[POST] @{autor_handle} ← '{texto[:70]}'")
        return True

    # ────────────────────────────────────────────────────────────
    # ENGAJAMENTO PROATIVO
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def comentar_post(self, texto: str, uri: str, cid: str) -> bool:
        """
        Comenta em um post público (engajamento proativo por termo de interesse).

        Args:
            texto: Texto do comentário.
            uri:   URI do post alvo.
            cid:   CID do post alvo.

        Returns:
            True se o comentário foi enviado.
        """
        if not self._rate.pode_postar():
            return False

        self._rate.aguardar_delay_minimo()

        parent_ref = models.ComAtprotoRepoStrongRef.Main(cid=cid, uri=uri)
        reply_ref = models.AppBskyFeedPost.ReplyRef(
            parent=parent_ref, root=parent_ref
        )
        self._client.send_post(
            text=texto[:LIMITE_CARACTERES_POST], reply_to=reply_ref
        )
        self._rate.registrar_post()
        logger.info(f"[PROATIVO] Comentou em {uri[:60]}")
        return True

    # ────────────────────────────────────────────────────────────
    # LEITURA DE FEED (APRENDIZADO)
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def obter_feed_home(self, limite: int = 30) -> List[str]:
        """
        Obtém textos do feed home para alimentar o corpus de aprendizado.

        Args:
            limite: Número máximo de posts a recuperar.

        Returns:
            Lista de textos de posts (sem filtragem de conteúdo — feita no brain).
        """
        textos: List[str] = []
        try:
            feed = self._client.app.bsky.feed.get_timeline({"limit": limite})
            for item in feed.feed:
                texto = getattr(item.post.record, "text", "")
                if texto and len(texto.strip()) > 20:
                    textos.append(texto.strip())
        except Exception as exc:
            logger.error(f"[FEED] Erro ao obter feed home: {exc}")
        logger.info(f"[FEED] {len(textos)} textos recuperados do feed home.")
        return textos

    @com_backoff
    def buscar_posts_por_termo(self, termo: str, limite: int = 10) -> List[Dict]:
        """
        Busca posts públicos contendo o termo especificado.

        Args:
            termo:  Termo de busca.
            limite: Número máximo de resultados.

        Returns:
            Lista de dicts com 'texto', 'uri', 'cid' e 'autor'.
        """
        resultados: List[Dict] = []
        try:
            resposta = self._client.app.bsky.feed.search_posts(
                {"q": termo, "limit": limite}
            )
            for post in resposta.posts:
                texto = getattr(post.record, "text", "")
                if texto and post.author.handle != BLUESKY_HANDLE:
                    resultados.append({
                        "texto": texto,
                        "uri": post.uri,
                        "cid": post.cid,
                        "autor": post.author.handle,
                    })
        except Exception as exc:
            logger.error(f"[BUSCA] Erro ao buscar termo '{termo}': {exc}")
        return resultados

    # ────────────────────────────────────────────────────────────
    # AUTO-FOLLOW
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def seguir_usuario(self, did: str, handle: str) -> bool:
        """
        Segue um usuário pelo DID.

        Args:
            did:    Identificador descentralizado do usuário.
            handle: Handle para logging.

        Returns:
            True se a operação foi bem-sucedida.
        """
        try:
            self._client.app.bsky.graph.follow.create(
                repo=self._client.me.did,
                record=models.AppBskyGraphFollow.Record(
                    subject=did,
                    created_at=self._client.get_current_time_iso(),
                ),
            )
            logger.info(f"[FOLLOW] Agora seguindo @{handle}")
            return True
        except Exception as exc:
            logger.warning(f"[FOLLOW] Não foi possível seguir @{handle}: {exc}")
            return False

    # ────────────────────────────────────────────────────────────
    # ATUALIZAÇÃO DE BIO
    # ────────────────────────────────────────────────────────────
    @com_backoff
    def atualizar_bio(self, nova_bio: str) -> bool:
        """
        Atualiza a biografia do perfil do Dimitri AI no Bluesky.

        Lê o perfil atual para preservar displayName e outros campos,
        alterando apenas a descrição.

        Args:
            nova_bio: Novo texto da bio (máx. 256 chars).

        Returns:
            True se a atualização foi bem-sucedida.
        """
        try:
            perfil = self._client.app.bsky.actor.get_profile(
                {"actor": BLUESKY_HANDLE}
            )
            display_name = getattr(perfil, "display_name", None) or "Dimitri AI"

            self._client.com.atproto.repo.put_record(
                models.ComAtprotoRepoPutRecord.Data(
                    repo=self._client.me.did,
                    collection="app.bsky.actor.profile",
                    rkey="self",
                    record={
                        "$type": "app.bsky.actor.profile",
                        "displayName": display_name,
                        "description": nova_bio[:256],
                    },
                )
            )
            logger.info(f"[BIO] Atualizada: '{nova_bio[:60]}'")
            return True
        except Exception as exc:
            logger.error(f"[BIO] Erro ao atualizar bio: {exc}")
            return False
