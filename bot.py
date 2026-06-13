import praw
import markovify
import time
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False
    print("Aviso: scikit-learn não está instalado. Usando similaridade simples.")

# ==========================================
# CONFIGURAÇÕES E DEPENDÊNCIAS
# ==========================================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words_pt = set(stopwords.words('portuguese'))

INTERVALO_POST = 7200
INTERVALO_COMENTARIO = 300
INTERVALO_VOTO = 60
ARQUIVO_HISTORICO = "comunidades_vistas.json"
ARQUIVO_MEMORIA = "memoria_dialogos.json"

reddit = praw.Reddit(
    client_id="SEU_CLIENT_ID",
    client_secret="SEU_CLIENT_SECRET",
    user_agent="script:minha_ia_bot:v1.0",
    username="NOME_DO_BOT",
    password="SENHA_DO_BOT"
)

# ==========================================
# CÉREBRO: LINGUAGEM E MEMÓRIA
# ==========================================
class CerebroConversacional:
    def __init__(self):
        self.memoria = self._carregar_memoria()
        self.modelo_markov = self._carregar_markov()

    def _carregar_memoria(self):
        if os.path.exists(ARQUIVO_MEMORIA):
            with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"entradas": [], "respostas": []}

    def _salvar_memoria(self):
        with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
            json.dump(self.memoria, f, ensure_ascii=False, indent=4)

    def _carregar_markov(self):
        try:
            with open("cerebro_inicial.txt", "r", encoding="utf-8") as f:
                return markovify.Text(f.read(), state_size=2)
        except FileNotFoundError:
            return None

    def extrair_palavras_chave(self, texto):
        palavras = word_tokenize(texto.lower())
        return [p for p in palavras if p.isalnum() and p not in stop_words_pt]

    def buscar_resposta_memoria(self, texto_usuario):
        if not self.memoria["entradas"]: return None
        vetorizador = TfidfVectorizer()
        vetores = vetorizador.fit_transform(self.memoria["entradas"] + [texto_usuario])
        similaridades = cosine_similarity(vetores[-1], vetores[:-1])
        indice_mais_similar = similaridades.argmax()
        
        if similaridades[0][indice_mais_similar] > 0.75:
            return self.memoria["respostas"][indice_mais_similar]
        return None

    def gerar_resposta_contextual(self, palavras_chave):
        if not self.modelo_markov: return "Processando dados..."
        for _ in range(500):
            frase = self.modelo_markov.make_short_sentence(280)
            if frase and any(p in frase.lower() for p in palavras_chave):
                return frase
        return self.modelo_markov.make_short_sentence(280)

    def processar_interacao(self, texto_usuario):
        resposta = self.buscar_resposta_memoria(texto_usuario)
        if not resposta:
            resposta = self.gerar_resposta_contextual(self.extrair_palavras_chave(texto_usuario))
            if resposta:
                self.memoria["entradas"].append(texto_usuario)
                self.memoria["respostas"].append(resposta)
                self._salvar_memoria()
                with open("cerebro_inicial.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n{texto_usuario}")
                self.modelo_markov = self._carregar_markov()
        return resposta

# ==========================================
# GERENCIADOR: AÇÕES NO REDDIT
# ==========================================
class GerenciadorBot:
    def __init__(self):
        self.humor = 0.0
        self.ultimo_post = 0
        self.ultimo_comentario = 0
        self.ultimo_voto = 0
        self.comunidades_vistas = self._carregar_historico()

    def _carregar_historico(self):
        if os.path.exists(ARQUIVO_HISTORICO):
            with open(ARQUIVO_HISTORICO, "r") as f: return json.load(f)
        return []

    def _salvar_historico(self):
        with open(ARQUIVO_HISTORICO, "w") as f: json.dump(self.comunidades_vistas, f)

    def atualizar_humor(self, texto):
        self.humor = (self.humor * 0.7) + (TextBlob(texto).sentiment.polarity * 0.3)
        return max(-1.0, min(1.0, self.humor))

    def verificar_primeira_interacao(self, subreddit_name):
        if subreddit_name not in self.comunidades_vistas:
            print(f"Postando aviso de IA no r/{subreddit_name}...")
            try:
                reddit.subreddit(subreddit_name).submit(
                    title="Olá! Sou uma Inteligência Artificial.",
                    selftext="Este perfil é um bot baseado em NLP aprendendo com a comunidade."
                )
                self.comunidades_vistas.append(subreddit_name)
                self._salvar_historico()
            except Exception as e:
                print(f"Erro no post inicial: {e}")

    def votar(self, alvo, sentimento):
        agora = time.time()
        if agora - self.ultimo_voto < INTERVALO_VOTO: return
        try:
            if sentimento > 0.3: alvo.upvote()
            elif sentimento < -0.3: alvo.downvote()
            self.ultimo_voto = agora
        except Exception: pass

    def postar_conteudo(self, subreddit_name, modelo_markov):
        agora = time.time()
        if agora - self.ultimo_post < INTERVALO_POST or not modelo_markov: return
        try:
            sub = reddit.subreddit(subreddit_name)
            flairs = list(sub.flair.link_templates)
            flair_id = flairs[0]['id'] if flairs else None
            sub.submit(title="Tópico Gerado", selftext=modelo_markov.make_long_sentence(500), flair_id=flair_id)
            self.ultimo_post = agora
        except Exception: pass

# ==========================================
# LOOP PRINCIPAL
# ==========================================
def executar_bot():
    print("Iniciando IA...")
    bot = GerenciadorBot()
    cerebro = CerebroConversacional()
    
    nome_sub = "test" # MUDAR AQUI PARA O SUBREDDIT DESEJADO
    subreddit = reddit.subreddit(nome_sub)
    
    bot.verificar_primeira_interacao(nome_sub)

    for comentario in subreddit.stream.comments(skip_existing=True):
        if comentario.author == reddit.user.me(): continue

        texto = comentario.body
        bot.votar(comentario, TextBlob(texto).sentiment.polarity)

        if "NOME_DO_BOT".lower() in texto.lower():
            agora = time.time()
            if agora - bot.ultimo_comentario >= INTERVALO_COMENTARIO:
                bot.atualizar_humor(texto)
                resposta = cerebro.processar_interacao(texto)
                
                if resposta:
                    comentario.reply(resposta)
                    print(f"Resposta gerada: {resposta}")
                    bot.ultimo_comentario = agora

        bot.postar_conteudo(nome_sub, cerebro.modelo_markov)

if __name__ == "__main__":
    executar_bot()