# 🚀 Guia de Deployment: Dimitri AI em Railway.app

Railway.app permite rodar apps Python **grátis** com até 5GB/mês de uso. Perfeito para um bot 24/7.

---

## 1️⃣ Preparação Local

### Pré-requisitos:
- Conta no GitHub (para hospedar o código)
- Conta no Railway.app (login com GitHub)
- Tokens já configurados (Bluesky + Hugging Face)

### Instruções:

**A) Criar repositório GitHub**
```bash
git init
git add .
git commit -m "Initial Dimitri AI commit"
git remote add origin https://github.com/seu_usuario/dimitri-ai.git
git branch -M main
git push -u origin main
```

⚠️ **IMPORTANTE**: Verifique que `.gitignore` tem `.env`!

```
# .gitignore
.env
.venv/
__pycache__/
logs/
*.pyc
```

---

## 2️⃣ Configurar Railway.app

### Passos:

1. **Acesse**: https://railway.app/
2. **Login** com GitHub
3. **New Project** → **Deploy from GitHub repo**
4. **Selecione** o repositório `dimitri-ai`
5. **Confirm Deploy**

---

## 3️⃣ Adicionar Variáveis de Ambiente

No dashboard do Railway:

1. Vá para **Variáveis** (Variables)
2. Adicione as seguintes:

```
BLUESKY_HANDLE=seu_handle.bsky.social
BLUESKY_APP_PASSWORD=sua_senha_de_app
ADMIN_HANDLE=seu_admin_handle.bsky.social
HUGGINGFACE_API_KEY=hf_sua_chave_aqui
```

3. **Deploy** (vai atualizar automaticamente)

---

## 4️⃣ Verificar Logs

1. No dashboard, clique no projeto
2. **Deployments** → **View Logs**
3. Procure por:
   - ✅ `Motor de IA: Qwen 0.5b`
   - ✅ `[LOGIN] Autenticado como @...`
   - ✅ `Dimitri AI está online`

---

## 5️⃣ Manutenção

### Atualizar código:
```bash
git add .
git commit -m "Meu update"
git push origin main
```
Railway detecta e redeploy automaticamente!

### Ver status:
- Railway Dashboard: Status do bot
- Bluesky: Procure por menções ao seu bot

### Parar o bot:
1. No Dashboard, **Environment** → **Remove**

---

## 💡 Dicas

- **Memory**: Railway aloca ~512MB grátis (suficiente para Qwen 0.5b)
- **Network**: API calls para Hugging Face → sem problema de conexão
- **Uptime**: ~99% em Railway.app
- **Custo**: 100% grátis com crédito de boas-vindas

---

## ⚠️ Troubleshooting

| Problema | Solução |
|----------|---------|
| "Credential not found" | Verifique as variáveis no Railway |
| "API rate limited" | Aguarde alguns minutos, Qwen tem rate limit de 5 req/min grátis |
| "Bot not responding" | Veja os logs, procure por erros |
| "Memory exceeded" | Reduza `MAX_MEMORIA` em `config.py` |

---

## 🔗 Links Úteis

- Railway Docs: https://docs.railway.app/
- GitHub: https://github.com/
- Bluesky: https://bsky.social/
- Hugging Face: https://huggingface.co/

---

Pronto! Seu bot estará rodando 24/7 sem precisar do seu PC. ✅
