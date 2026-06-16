"""
config.py — Configurações centrais do Dimitri AI.

NUNCA coloque credenciais aqui diretamente.
Use um arquivo .env na raiz do projeto (veja .env.example).
"""
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ================================================================
# AUTENTICAÇÃO BLUESKY (via variáveis de ambiente)
# ================================================================
BLUESKY_HANDLE: str = os.getenv("BLUESKY_HANDLE", "dimitriai.bsky.social").strip()
if BLUESKY_HANDLE.startswith("@"):  # Normaliza caso o usuário tenha incluído o @
    BLUESKY_HANDLE = BLUESKY_HANDLE[1:]
BLUESKY_APP_PASSWORD: str = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
ADMIN_HANDLE: str = os.getenv("ADMIN_HANDLE", "").strip()


def validar_credenciais_bluesky() -> None:
    """Valida se as credenciais do Bluesky foram carregadas corretamente."""
    if not BLUESKY_HANDLE or not BLUESKY_HANDLE.strip():
        raise ValueError(
            "BLUESKY_HANDLE não encontrado. Crie um arquivo .env a partir de .env.example "
            "e preencha a variável BLUESKY_HANDLE."
        )
    if not BLUESKY_APP_PASSWORD or not BLUESKY_APP_PASSWORD.strip():
        raise ValueError(
            "BLUESKY_APP_PASSWORD não encontrado. Crie um arquivo .env a partir de .env.example "
            "e preencha a variável BLUESKY_APP_PASSWORD."
        )

# ================================================================
# DIRETÓRIOS E ARQUIVOS
# ================================================================
DATA_DIR: str = "data"
LOG_DIR: str = "logs"
ARQUIVO_MEMORIA: str = os.path.join(DATA_DIR, "memoria_dialogos.json")
ARQUIVO_CEREBRO: str = os.path.join(DATA_DIR, "cerebro_inicial.txt")
ARQUIVO_LOG: str = os.path.join(LOG_DIR, "dimitri.log")

# ================================================================
# INTERVALOS DE ROTINAS (em segundos)
# ================================================================
INTERVALO_NOTIFICACOES: int = 30       # Verificação de menções
INTERVALO_FEED: int = 10              # Aprendizado via feed home (10 min)
INTERVALO_BUSCA_TERMOS: int = 1800     # Engajamento proativo (30 min)
INTERVALO_BIO_UPDATE: int = 3600       # Atualização de bio (1 hora)

# ================================================================
# PARÂMETROS DO CÉREBRO
# ================================================================
MAX_MEMORIA: int = 5000                # Máx de pares entrada/resposta
MAX_TENTATIVAS_MARKOV: int = 500       # Tentativas de geração contextual
LIMITE_CARACTERES_POST: int = 295      # Limite seguro de chars no Bluesky
SIMILARIDADE_MINIMA: float = 0.75     # Threshold de similaridade TF-IDF
TAMANHO_CONTEXTO_THREAD: int = 5      # Quantas trocas anteriores guardar por thread
MARKOV_STATE_SIZE: int = 2             # Tamanho do estado da cadeia de Markov

# ================================================================
# RATE LIMITING
# ================================================================
MAX_POSTS_POR_HORA: int = 20          # Máx de posts enviados por hora
DELAY_ENTRE_POSTS: float = 3.5        # Delay mínimo entre posts (segundos)
MAX_ENGAJAMENTOS_POR_CICLO: int = 3   # Máx de comentários proativos por rodada

# ================================================================
# EXPONENTIAL BACKOFF
# ================================================================
BACKOFF_BASE: float = 2.0             # Base exponencial
BACKOFF_MAX: float = 300.0            # Espera máxima (5 minutos)
BACKOFF_TENTATIVAS: int = 5           # Tentativas antes de desistir

# ================================================================
# ADMINISTRAÇÃO
# ================================================================
ADMIN_HANDLE: str = os.getenv("ADMIN_HANDLE", "")
COMANDOS_ADMIN: dict = {
    "!limpar_memoria": "clear_memory",
    "!recarregar":     "reload_model",
    "!status":         "show_status",
}

# ================================================================
# TERMOS DE INTERESSE PARA ENGAJAMENTO PROATIVO
# ================================================================
TERMOS_INTERESSE: List[str] = [
    "inteligência artificial",
    "machine learning",
    "python",
    "programação",
    "chatbot",
    "brawl stars",
    "deep learning",
    "tecnologia",
]

# ================================================================
# FILTROS DE CONTEÚDO
# ================================================================
# Palavras proibidas — o bot não aprende nem reproduz essas palavras
PALAVRAS_PROIBIDAS: List[str] = [
    "merda", "porra", "caralho", "puta", "viado", "fdp",
    "imbecil", "idiota", "burro", "cuzão",
]

# Padrões regex que identificam spam ou conteúdo indesejado
PADROES_SPAM: List[str] = [
    r"https?://\S+",       # Links
    r"(.)\1{5,}",          # Caracteres repetidos 5+ vezes (ex: "aaaaa")
    r"@\w+\s+@\w+\s+@\w+", # Menção em cadeia (spam de @)
    r"\b\d{5,}\b",          # Números longos (possível código/scam)
]

# Comprimento mínimo e máximo de texto válido
TEXTO_MIN_CHARS: int = 3
TEXTO_MAX_CHARS: int = 1000
