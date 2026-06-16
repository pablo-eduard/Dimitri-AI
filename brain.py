"""
brain.py — Núcleo de inteligência do Dimitri AI.

Responsabilidades:
  - Memória de longo prazo (arquivo JSON)
  - Memória de curto prazo por thread (deque em memória)
  - Geração de texto via Cadeia de Markov
  - Busca semântica por TF-IDF + Similaridade de Cosseno
  - Extração de palavras-chave com stemming RSLP (português)
  - Sistema de fallback hierárquico inteligente
"""
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import markovify

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False

from config import (
    ARQUIVO_MEMORIA, ARQUIVO_CEREBRO,
    MAX_MEMORIA, MAX_TENTATIVAS_MARKOV,
    SIMILARIDADE_MINIMA, MARKOV_STATE_SIZE,
    TAMANHO_CONTEXTO_THREAD,
)
from utils.logger import logger
from utils.sanitizer import sanitizar_saida, higienizar_para_cerebro

# Baixa recursos NLTK necessários silenciosamente
for _recurso in ("punkt", "punkt_tab", "stopwords", "rslp"):
    nltk.download(_recurso, quiet=True)


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
# CÉREBRO CONVERSACIONAL
# ════════════════════════════════════════════════════════════════
class CerebroConversacional:
    """
    Orquestra todo o pipeline de geração de resposta do Dimitri AI.

    Fluxo de processamento:
      1. Sanitização da entrada (no chamador)
      2. Busca semântica na memória de longo prazo (TF-IDF)
      3. Geração Markov contextual (palavras-chave + stemming)
      4. Fallback hierárquico inteligente
      5. Aprendizado com a nova interação
    """

    # Templates de fallback por categoria semântica
    _FALLBACKS: Dict[str, List[str]] = {
        "saudacao": [
            "Olá! O Dimitri AI está em pleno funcionamento. O que quer explorar hoje?",
            "Oi! Estou aqui aprendendo com cada conversa. Me conta o que você pensa!",
        ],
        "programacao": [
            "Código é a linguagem que mais me fascina. O que você está construindo?",
            "Python, IA, sistemas... adoro esses temas! Pode elaborar mais?",
        ],
        "jogo": [
            "Jogos são laboratórios de sistemas complexos. Qual você está explorando?",
            "A IA em jogos é incrível! Me conta mais sobre o que você está jogando.",
        ],
        "pergunta": [
            "Ótima pergunta! Meu modelo está processando... Me ajuda com mais detalhes?",
            "Isso é interessante. Você pode me dar um exemplo do que quer dizer?",
        ],
        "default": [
            "Cada mensagem alimenta meu aprendizado. Me conta mais!",
            "Processando... Meu cérebro de Markov está a todo vapor. Elabora mais?",
            "Interessante perspectiva. Estou evoluindo com essa informação.",
        ],
    }

    def __init__(self) -> None:
        self._stemmer = RSLPStemmer()
        self._stop_words = set(stopwords.words("portuguese"))
        self.memoria = self._carregar_memoria()
        self.modelo_markov = self._carregar_markov()
        self.memoria_curto = MemoriaCurtoPrazo()
        self._vetorizador: Optional[TfidfVectorizer] = None
        self._cache_vetores = None
        self._reconstruir_tfidf()
        logger.info(
            f"Cérebro inicializado | Memórias: {len(self.memoria['entradas'])} | "
            f"Markov: {'OK' if self.modelo_markov else 'INATIVO'} | "
            f"TF-IDF: {'OK' if SKLEARN_DISPONIVEL else 'INDISPONÍVEL'}"
        )

    # ────────────────────────────────────────────────────────────
    # MEMÓRIA DE LONGO PRAZO
    # ────────────────────────────────────────────────────────────
    def _carregar_memoria(self) -> Dict:
        """Carrega o arquivo JSON de memória, criando um novo se necessário."""
        if os.path.exists(ARQUIVO_MEMORIA):
            try:
                with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
                    dados = json.load(f)
                # Validação de estrutura mínima
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
                # Mantém apenas as N interações mais recentes
                self.memoria["entradas"] = self.memoria["entradas"][-MAX_MEMORIA:]
                self.memoria["respostas"] = self.memoria["respostas"][-MAX_MEMORIA:]
                logger.info(f"Memória podada para os {MAX_MEMORIA} pares mais recentes.")
            with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
                json.dump(self.memoria, f, ensure_ascii=False, indent=4)
        except IOError as exc:
            logger.error(f"Erro ao salvar memória: {exc}")

    def limpar_memoria(self) -> None:
        """Apaga toda a memória de longo prazo (comando de admin)."""
        self.memoria = {"entradas": [], "respostas": []}
        self._salvar_memoria()
        self._vetorizador = None
        self._cache_vetores = None
        logger.info("Memória de longo prazo apagada pelo administrador.")

    # ────────────────────────────────────────────────────────────
    # MODELO MARKOV
    # ────────────────────────────────────────────────────────────
    def _carregar_markov(self) -> Optional[markovify.Text]:
        """Carrega ou recarrega o modelo Markov a partir do cerebro_inicial.txt."""
        try:
            with open(ARQUIVO_CEREBRO, "r", encoding="utf-8") as f:
                conteudo = f.read().strip()
            if len(conteudo) < 50:
                logger.warning("cerebro_inicial.txt muito pequeno para Markov.")
                return None
            modelo = markovify.Text(conteudo, state_size=MARKOV_STATE_SIZE)
            logger.debug("Modelo Markov carregado/atualizado.")
            return modelo
        except FileNotFoundError:
            logger.error(f"'{ARQUIVO_CEREBRO}' não encontrado.")
            return None
        except Exception as exc:
            logger.error(f"Erro ao carregar Markov: {exc}")
            return None

    def recarregar_modelo(self) -> None:
        """Recarrega Markov e TF-IDF (usado após injeções em lote ou comando admin)."""
        self.modelo_markov = self._carregar_markov()
        self._reconstruir_tfidf()
        logger.info("Modelo Markov e cache TF-IDF recarregados.")

    def injetar_no_cerebro(self, texto: str) -> bool:
        """
        Injeta texto higienizado no cerebro_inicial.txt e atualiza o modelo.

        Args:
            texto: Texto a ser adicionado ao corpus de treinamento.

        Returns:
            True se a injeção foi bem-sucedida, False caso contrário.
        """
        texto_limpo = higienizar_para_cerebro(texto)
        if not texto_limpo:
            return False
        try:
            with open(ARQUIVO_CEREBRO, "a", encoding="utf-8") as f:
                f.write(f"\n{texto_limpo}")
            # Recarrega o modelo apenas após injeção (custo pontual)
            self.modelo_markov = self._carregar_markov()
            logger.debug(f"Injeção no cérebro: '{texto_limpo[:70]}'")
            return True
        except IOError as exc:
            logger.error(f"Falha na injeção no cérebro: {exc}")
            return False

    # ────────────────────────────────────────────────────────────
    # TF-IDF / BUSCA SEMÂNTICA
    # ────────────────────────────────────────────────────────────
    def _reconstruir_tfidf(self) -> None:
        """Reconstrói o cache TF-IDF a partir das entradas atuais da memória."""
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
        Busca a resposta semanticamente mais próxima na memória de longo prazo.

        Usa TF-IDF vetorizado em cache (O(1) em buscas após construção).

        Args:
            texto: Texto do usuário para comparar.

        Returns:
            Resposta memorizada, ou None se abaixo do threshold de similaridade.
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
        """
        Extrai palavras-chave via tokenização NLTK + remoção de stop words + stemming RSLP.

        O stemming reduz as palavras à sua raiz, aumentando a cobertura semântica
        na busca por contexto no Markov.

        Args:
            texto: Texto para extração.

        Returns:
            Lista de stems únicos das palavras-chave.
        """
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
            # Fallback simples sem NLTK
            return [w.lower() for w in texto.split() if len(w) > 3][:8]

    # ────────────────────────────────────────────────────────────
    # GERAÇÃO DE RESPOSTA
    # ────────────────────────────────────────────────────────────
    def _gerar_markov(self, palavras_chave: List[str]) -> Optional[str]:
        """
        Tenta gerar uma frase contextualizada com o modelo Markov.

        Estratégia:
          1. Tentativa com palavras-chave (contextual)
          2. Fallback para frase genérica do modelo

        Args:
            palavras_chave: Lista de stems para guiar a busca contextual.

        Returns:
            Frase gerada ou None se o modelo estiver inativo.
        """
        if not self.modelo_markov:
            return None

        # Tentativa contextual
        for _ in range(MAX_TENTATIVAS_MARKOV):
            try:
                frase = self.modelo_markov.make_short_sentence(
                    max_chars=250, min_chars=20, tries=5
                )
                if frase and any(p in frase.lower() for p in palavras_chave):
                    return frase
            except Exception:
                break

        # Fallback: qualquer frase do modelo
        try:
            return self.modelo_markov.make_short_sentence(max_chars=250, min_chars=20)
        except Exception:
            return None

    def _fallback_hierarquico(
        self, palavras_chave: List[str], contexto_thread: List[Dict]
    ) -> str:
        """
        Sistema de fallback em cascata quando Markov falha ou retorna None.

        Nível 1 — Continuidade de thread: usa a última resposta da conversa.
        Nível 2 — Template semântico: detecta o tema e escolhe um template adequado.

        Args:
            palavras_chave: Stems da mensagem atual.
            contexto_thread: Histórico recente da thread.

        Returns:
            Resposta de fallback não genérica.
        """
        # Nível 1: usa o contexto da thread para dar continuidade
        if contexto_thread:
            ultima = contexto_thread[-1]
            trecho = ultima["resposta"][:100]
            continuacoes = [
                f"Como eu comentei: '{trecho}...' O que mais você quer explorar?",
                f"Retomando o que discutimos... Me conta mais sobre isso!",
            ]
            return random.choice(continuacoes)

        # Nível 2: detecta a categoria semântica pelo conjunto de stems
        chaves_lower = " ".join(palavras_chave)
        if any(s in chaves_lower for s in ["ola", "oi", "eae", "tud", "bem"]):
            categoria = "saudacao"
        elif any(s in chaves_lower for s in ["pyt", "cod", "program", "ia", "intelig", "script"]):
            categoria = "programacao"
        elif any(s in chaves_lower for s in ["jog", "game", "brawl", "play", "rpg"]):
            categoria = "jogo"
        elif "?" in chaves_lower or any(s in chaves_lower for s in ["qued", "com", "qual", "quem"]):
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
        Pipeline completo de geração de resposta.

        Etapas:
          1. Busca semântica na memória (TF-IDF)
          2. Geração Markov contextual
          3. Fallback hierárquico inteligente
          4. Aprendizado com a nova interação
          5. Registro na memória de curto prazo

        Args:
            texto_usuario:    Texto sanitizado do usuário.
            thread_uri:       URI da thread (para memória de curto prazo).
            contexto_externo: Contexto extra de pesquisa web (RAG).

        Returns:
            Resposta sanitizada e pronta para postagem.
        """
        contexto_thread = self.memoria_curto.obter(thread_uri) if thread_uri else []

        # Enriquece a query com contexto externo (RAG simples)
        query = f"{texto_usuario} {contexto_externo}" if contexto_externo else texto_usuario

        # ── Etapa 1: Memória semântica ────────────────────────────
        resposta = self.buscar_na_memoria(query)

        if not resposta:
            # ── Etapa 2: Geração Markov ───────────────────────────
            palavras_chave = self.extrair_palavras_chave(query)
            resposta = self._gerar_markov(palavras_chave)

            if not resposta:
                # ── Etapa 3: Fallback inteligente ─────────────────
                resposta = self._fallback_hierarquico(palavras_chave, contexto_thread)

            # ── Etapa 4: Aprendizado com a nova interação ─────────
            self.memoria["entradas"].append(texto_usuario)
            self.memoria["respostas"].append(resposta)
            self._salvar_memoria()
            self._reconstruir_tfidf()
            self.injetar_no_cerebro(texto_usuario)

        # ── Etapa 5: Memória de curto prazo ──────────────────────
        if thread_uri:
            self.memoria_curto.adicionar(thread_uri, texto_usuario, resposta)

        return sanitizar_saida(resposta)

    def obter_status(self) -> Dict:
        """
        Retorna um snapshot do estado atual do cérebro.
        Usado para logs de status e atualização de bio.
        """
        return {
            "total_memorias": len(self.memoria["entradas"]),
            "modelo_markov_ativo": self.modelo_markov is not None,
            "tfidf_disponivel": SKLEARN_DISPONIVEL and self._vetorizador is not None,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
