"""
brain_qwen.py — Cérebro conversacional com Qwen 0.5b via Hugging Face API.

Responsabilidades:
  - Memória de longo prazo (arquivo JSON) — igual ao original
  - Memória de curto prazo por thread (deque em memória)
  - Geração de texto via Qwen 0.5b (chamada HTTP à HF API)
  - Fallback inteligente quando API falha
  - Busca semântica básica com TF-IDF (opcional, mantém compatibilidade)
"""
import os
import json
import random
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False

from config import (
    ARQUIVO_MEMORIA, ARQUIVO_CEREBRO,
    MAX_MEMORIA, SIMILARIDADE_MINIMA,
    TAMANHO_CONTEXTO_THREAD,
)
from utils.logger import logger
from utils.sanitizer import sanitizar_saida, higienizar_para_cerebro

# Baixa recursos NLTK necessários silenciosamente
for _recurso in ("punkt", "punkt_tab", "stopwords", "rslp"):
    nltk.download(_recurso, quiet=True)

# ════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES QWEN
# ════════════════════════════════════════════════════════════════
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MODELO_QWEN = "Qwen/Qwen0.5B"  # Modelo leve e rápido
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODELO_QWEN}"

# Parâmetros de geração
QWEN_TEMPERATURE = 0.7
QWEN_MAX_TOKENS = 150
QWEN_TOP_P = 0.9
QWEN_RETRY_ATTEMPTS = 3
QWEN_RETRY_DELAY = 2  # segundos


# ════════════════════════════════════════════════════════════════
# MEMÓRIA DE CURTO PRAZO (por thread)
# ════════════════════════════════════════════════════════════════
class MemoriaCurtoPrazo:
    """
    Mantém o contexto recente de cada thread de conversa.
    Usa deque para limitar automaticamente o tamanho por thread.
    """

    def __init__(self, tamanho_max: int = TAMANHO_CONTEXTO_THREAD):
        self._contextos: Dict[str, deque] = {}
        self._tamanho_max = tamanho_max

    def adicionar(self, thread_uri: str, entrada: str, resposta: str) -> None:
        """Registra uma nova troca em uma thread."""
        if thread_uri not in self._contextos:
            self._contextos[thread_uri] = deque(maxlen=self._tamanho_max)
        self._contextos[thread_uri].append({"entrada": entrada, "resposta": resposta})

    def obter(self, thread_uri: str) -> List[Dict[str, str]]:
        """Retorna o histórico recente da thread."""
        return list(self._contextos.get(thread_uri, []))

    def limpar(self, thread_uri: str) -> None:
        """Remove o contexto de uma thread específica."""
        self._contextos.pop(thread_uri, None)


# ════════════════════════════════════════════════════════════════
# CLIENTE QWEN VIA HUGGING FACE API
# ════════════════════════════════════════════════════════════════
class ClienteQwen:
    """
    Wrapper para chamadas à API do Hugging Face com retry automático.
    """

    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        }
        self._api_disponivel = bool(HUGGINGFACE_API_KEY)

    def gerar_resposta(self, prompt: str) -> Optional[str]:
        """
        Gera uma resposta usando Qwen 0.5b via HF API.

        Args:
            prompt: Texto para o modelo completar.

        Returns:
            Texto gerado ou None se falhar após retries.
        """
        if not self._api_disponivel:
            logger.warning("HUGGINGFACE_API_KEY não configurada. Usando fallback.")
            return None

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": QWEN_MAX_TOKENS,
                "temperature": QWEN_TEMPERATURE,
                "top_p": QWEN_TOP_P,
                "do_sample": True,
            },
        }

        for tentativa in range(1, QWEN_RETRY_ATTEMPTS + 1):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()

                resultado = response.json()
                if isinstance(resultado, list) and len(resultado) > 0:
                    texto = resultado[0].get("generated_text", "")
                    # Remove o prompt do início da resposta gerada
                    if texto.startswith(prompt):
                        texto = texto[len(prompt):].strip()
                    if texto:
                        logger.debug(f"[QWEN] Gerou: '{texto[:60]}'")
                        return texto

                logger.warning(f"[QWEN] Resposta vazia: {resultado}")
                return None

            except requests.exceptions.Timeout:
                logger.warning(f"[QWEN] Timeout na tentativa {tentativa}/{QWEN_RETRY_ATTEMPTS}")
                if tentativa < QWEN_RETRY_ATTEMPTS:
                    time.sleep(QWEN_RETRY_DELAY)

            except requests.exceptions.HTTPError as exc:
                if response.status_code == 503:
                    logger.warning(f"[QWEN] Modelo carregando (503)... tentativa {tentativa}/{QWEN_RETRY_ATTEMPTS}")
                    time.sleep(QWEN_RETRY_DELAY * 2)
                else:
                    logger.error(f"[QWEN] Erro HTTP {response.status_code}: {exc}")
                    return None

            except requests.exceptions.RequestException as exc:
                logger.error(f"[QWEN] Erro de requisição: {exc}")
                if tentativa < QWEN_RETRY_ATTEMPTS:
                    time.sleep(QWEN_RETRY_DELAY)

            except Exception as exc:
                logger.error(f"[QWEN] Erro inesperado: {exc}")
                return None

        logger.error(f"[QWEN] Falhou após {QWEN_RETRY_ATTEMPTS} tentativas.")
        return None


# ════════════════════════════════════════════════════════════════
# CÉREBRO CONVERSACIONAL COM QWEN
# ════════════════════════════════════════════════════════════════
class CerebroConversacional:
    """
    Orquestra todo o pipeline com Qwen 0.5b.

    Fluxo de processamento:
      1. Sanitização da entrada (no chamador)
      2. Busca semântica na memória (TF-IDF, opcional)
      3. Geração com Qwen via API HF
      4. Fallback hierárquico inteligente
      5. Aprendizado com a nova interação
    """

    # Templates de fallback por categoria
    _FALLBACKS: Dict[str, List[str]] = {
        "saudacao": [
            "Olá! Tudo bem com você? Dimitri AI aqui, pronto para conversar!",
            "Oi! Que bom te ver. O que está em sua mente hoje?",
        ],
        "programacao": [
            "Código é fascinante! Qual linguagem ou projeto você está explorando?",
            "Python, IA, sistemas... adoro esses tópicos. Me conta mais!",
        ],
        "jogo": [
            "Jogos são laboratórios de lógica complexa. Qual você está jogando?",
            "A IA em jogos é incrível! Qual é seu favorito?",
        ],
        "pergunta": [
            "Ótima pergunta! Pode dar mais detalhes ou um exemplo?",
            "Isso é interessante. Explora mais esse pensamento!",
        ],
        "default": [
            "Cada conversa me ajuda a aprender. Me conta mais sobre isso!",
            "Interessante perspectiva. Continua elaborando!",
            "Fascinante! Qual é o contexto disso?",
        ],
    }

    def __init__(self) -> None:
        self._stemmer = RSLPStemmer()
        self._stop_words = set(stopwords.words("portuguese"))
        self.cliente_qwen = ClienteQwen()
        self.memoria = self._carregar_memoria()
        self.memoria_curto = MemoriaCurtoPrazo()
        self._vetorizador: Optional[TfidfVectorizer] = None
        self._cache_vetores = None
        self._reconstruir_tfidf()
        
        status = (
            f"Cérebro inicializado | Memórias: {len(self.memoria['entradas'])} | "
            f"Qwen: {'ATIVO' if self.cliente_qwen._api_disponivel else 'INATIVO'} | "
            f"TF-IDF: {'OK' if SKLEARN_DISPONIVEL else 'INDISPONÍVEL'}"
        )
        logger.info(status)

    # ────────────────────────────────────────────────────────────
    # MEMÓRIA DE LONGO PRAZO
    # ────────────────────────────────────────────────────────────
    def _carregar_memoria(self) -> Dict:
        """Carrega o arquivo JSON de memória, criando um novo se necessário."""
        if os.path.exists(ARQUIVO_MEMORIA):
            try:
                with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
                    dados = json.load(f)
                if "entradas" not in dados or "respostas" not in dados:
                    raise ValueError("Estrutura inválida.")
                logger.info(f"Memória carregada: {len(dados['entradas'])} pares.")
                return dados
            except (json.JSONDecodeError, ValueError, IOError) as exc:
                logger.error(f"Erro ao carregar memória ({exc}). Criando nova.")
        return {"entradas": [], "respostas": []}

    def _salvar_memoria(self) -> None:
        """Persiste a memória em disco, podando se necessário."""
        try:
            if len(self.memoria["entradas"]) > MAX_MEMORIA:
                self.memoria["entradas"] = self.memoria["entradas"][-MAX_MEMORIA:]
                self.memoria["respostas"] = self.memoria["respostas"][-MAX_MEMORIA:]
                logger.info(f"Memória podada para os {MAX_MEMORIA} pares mais recentes.")
            with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
                json.dump(self.memoria, f, ensure_ascii=False, indent=4)
        except IOError as exc:
            logger.error(f"Erro ao salvar memória: {exc}")

    def limpar_memoria(self) -> None:
        """Apaga toda a memória de longo prazo."""
        self.memoria = {"entradas": [], "respostas": []}
        self._salvar_memoria()
        self._vetorizador = None
        self._cache_vetores = None
        logger.info("Memória de longo prazo apagada pelo administrador.")

    # ────────────────────────────────────────────────────────────
    # TF-IDF / BUSCA SEMÂNTICA
    # ────────────────────────────────────────────────────────────
    def _reconstruir_tfidf(self) -> None:
        """Reconstrói o cache TF-IDF a partir das entradas atuais."""
        if not SKLEARN_DISPONIVEL or not self.memoria["entradas"]:
            return
        try:
            self._vetorizador = TfidfVectorizer(
                min_df=1, ngram_range=(1, 2), sublinear_tf=True
            )
            self._cache_vetores = self._vetorizador.fit_transform(
                self.memoria["entradas"]
            )
            logger.debug(
                f"Cache TF-IDF reconstruído com {len(self.memoria['entradas'])} entradas."
            )
        except Exception as exc:
            logger.error(f"Erro ao construir TF-IDF: {exc}")
            self._vetorizador = None
            self._cache_vetores = None

    def buscar_na_memoria(self, texto: str) -> Optional[str]:
        """
        Busca a resposta semanticamente mais próxima na memória.
        """
        if (
            not SKLEARN_DISPONIVEL
            or self._vetorizador is None
            or self._cache_vetores is None
        ):
            return None
        try:
            vetor_query = self._vetorizador.transform([texto])
            similaridades = cosine_similarity(vetor_query, self._cache_vetores)
            idx = int(similaridades.argmax())
            score = float(similaridades[0][idx])
            if score >= SIMILARIDADE_MINIMA:
                logger.debug(
                    f"Match semântico (score={score:.3f}): "
                    f"'{self.memoria['entradas'][idx][:40]}'"
                )
                return self.memoria["respostas"][idx]
        except Exception as exc:
            logger.error(f"Erro na busca semântica: {exc}")
        return None

    # ────────────────────────────────────────────────────────────
    # EXTRAÇÃO DE PALAVRAS-CHAVE
    # ────────────────────────────────────────────────────────────
    def extrair_palavras_chave(self, texto: str) -> List[str]:
        """Extrai palavras-chave via tokenização NLTK + stemming."""
        try:
            tokens = word_tokenize(texto.lower(), language="portuguese")
            stems = [
                self._stemmer.stem(token)
                for token in tokens
                if token.isalpha()
                and token not in self._stop_words
                and len(token) > 2
            ]
            return list(set(stems))
        except Exception as exc:
            logger.error(f"Erro na extração de palavras-chave: {exc}")
            return [w.lower() for w in texto.split() if len(w) > 3][:8]

    # ────────────────────────────────────────────────────────────
    # GERAÇÃO COM QWEN
    # ────────────────────────────────────────────────────────────
    def _gerar_qwen(self, texto_usuario: str, contexto_thread: List[Dict]) -> Optional[str]:
        """
        Gera resposta usando Qwen 0.5b via Hugging Face API.

        Args:
            texto_usuario: Texto do usuário.
            contexto_thread: Histórico da thread para contexto.

        Returns:
            Resposta gerada ou None.
        """
        # Constrói prompt com contexto da thread
        prompt = texto_usuario
        if contexto_thread:
            historico = "\n".join(
                f"- Entrada: {t['entrada']}\n  Resposta: {t['resposta']}"
                for t in contexto_thread[-2:]  # Últimas 2 trocas
            )
            prompt = f"Contexto recente:\n{historico}\n\nNova entrada: {texto_usuario}\n\nResposta:"

        resposta = self.cliente_qwen.gerar_resposta(prompt)
        return resposta

    # ────────────────────────────────────────────────────────────
    # FALLBACK HIERÁRQUICO
    # ────────────────────────────────────────────────────────────
    def _fallback_hierarquico(
        self, palavras_chave: List[str], contexto_thread: List[Dict]
    ) -> str:
        """
        Sistema de fallback em cascata quando Qwen falha.
        """
        if contexto_thread:
            ultima = contexto_thread[-1]
            trecho = ultima["resposta"][:100]
            continuacoes = [
                f"Como eu comentei: '{trecho}...' O que mais você quer explorar?",
                f"Retomando o que discutimos... Me conta mais sobre isso!",
            ]
            return random.choice(continuacoes)

        # Detecta categoria semântica
        chaves_lower = " ".join(palavras_chave)
        if any(s in chaves_lower for s in ["ola", "oi", "eae", "tud", "bem"]):
            categoria = "saudacao"
        elif any(s in chaves_lower for s in ["pyt", "cod", "program", "ia", "intelig"]):
            categoria = "programacao"
        elif any(s in chaves_lower for s in ["jog", "game", "brawl", "play", "rpg"]):
            categoria = "jogo"
        elif "?" in chaves_lower or any(s in chaves_lower for s in ["qued", "com", "qual"]):
            categoria = "pergunta"
        else:
            categoria = "default"

        return random.choice(self._FALLBACKS[categoria])

    # ────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ────────────────────────────────────────────────────────────
    def processar_interacao(
        self,
        texto_usuario: str,
        thread_uri: Optional[str] = None,
        contexto_externo: Optional[str] = None,
    ) -> str:
        """
        Pipeline completo de geração de resposta com Qwen.

        Etapas:
          1. Busca semântica na memória (TF-IDF)
          2. Geração com Qwen via API
          3. Fallback hierárquico inteligente
          4. Aprendizado com a nova interação
          5. Registro na memória de curto prazo
        """
        contexto_thread = self.memoria_curto.obter(thread_uri) if thread_uri else []

        # Enriquece a query com contexto externo (RAG simples)
        query = f"{texto_usuario} {contexto_externo}" if contexto_externo else texto_usuario

        # ── Etapa 1: Memória semântica ────────────────────────────
        resposta = self.buscar_na_memoria(query)

        if not resposta:
            # ── Etapa 2: Geração com Qwen ────────────────────────
            resposta = self._gerar_qwen(texto_usuario, contexto_thread)

            if not resposta:
                # ── Etapa 3: Fallback inteligente ─────────────────
                palavras_chave = self.extrair_palavras_chave(query)
                resposta = self._fallback_hierarquico(palavras_chave, contexto_thread)

            # ── Etapa 4: Aprendizado com a nova interação ─────────
            self.memoria["entradas"].append(texto_usuario)
            self.memoria["respostas"].append(resposta)
            self._salvar_memoria()
            self._reconstruir_tfidf()

        # ── Etapa 5: Memória de curto prazo ──────────────────────
        if thread_uri:
            self.memoria_curto.adicionar(thread_uri, texto_usuario, resposta)

        return sanitizar_saida(resposta)

    def obter_status(self) -> Dict:
        """Retorna um snapshot do estado atual do cérebro."""
        return {
            "total_memorias": len(self.memoria["entradas"]),
            "qwen_disponivel": self.cliente_qwen._api_disponivel,
            "tfidf_disponivel": SKLEARN_DISPONIVEL and self._vetorizador is not None,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
