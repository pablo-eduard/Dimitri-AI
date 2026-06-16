# 🤖 Dimitri AI com Qwen 0.5b

Adaptação completa do Dimitri AI para usar **Qwen 0.5b** via **Hugging Face API** com suporte para deployment em **Railway.app** (24/7, online).

---

## 📋 Arquivos Modificados/Novos

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `brain_qwen.py` | ✅ NOVO | Cérebro com Qwen 0.5b via API HF |
| `main.py` | 🔄 ATUALIZADO | Seleciona brain_qwen ou brain automaticamente |
| `requirements.txt` | 🔄 ATUALIZADO | Adicionado `requests` |
| `.env.example` | 🔄 ATUALIZADO | Adicionada variável `HUGGINGFACE_API_KEY` |
| `Dockerfile` | ✅ NOVO | Para deployment em Railway |
| `DEPLOYMENT_RAILWAY.md` | ✅ NOVO | Guia passo-a-passo |
| `dimitri_colab.ipynb` | ✅ NOVO | Notebook para testar no Google Colab |

---

## 🎯 Como Funciona?

### Fluxo de Processamento

```
Mensagem do Usuário
    ↓
[main.py] Verifica HUGGINGFACE_API_KEY
    ├─ Se existe → Usa brain_qwen.py (Qwen via API)
    └─ Se não → Usa brain.py (Markov local)
    ↓
[brain_qwen.py] Pipeline:
    1. Busca semântica na memória (TF-IDF)
    2. Se não encontra → Chama Qwen 0.5b via API HF
    3. Se API falha → Fallback inteligente
    4. Aprende com a interação (salva em JSON)
    ↓
[bluesky_client.py] Posta resposta no Bluesky
```

### Vantagens do Qwen 0.5b

- ✅ **Leve**: Só 500MB (cabe em CPU)
- ✅ **Rápido**: <2s por resposta mesmo em CPU
- ✅ **Multilíngue**: Português, inglês, chinês
- ✅ **Grátis**: Via Hugging Face Inference API

---

## 🚀 Quick Start

### 1. Obter Chaves

**Bluesky App Password:**
- Vá para: https://bsky.app/settings/app-passwords
- Crie uma nova (marque ✅ "Moderate reports", ✅ "Access your feeds")
- Copie a senha

**Hugging Face API Key:**
- Vá para: https://huggingface.co/settings/tokens
- Crie novo token com permissão "Read"
- Copie a chave

### 2. Configurar `.env`

```bash
cp .env.example .env
```

Preencha com seus dados:

```env
BLUESKY_HANDLE=seu_handle.bsky.social
BLUESKY_APP_PASSWORD=sua_senha
ADMIN_HANDLE=seu_admin.bsky.social
HUGGINGFACE_API_KEY=hf_sua_chave_aqui
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Rodar Localmente

```bash
python main.py
```

Procure no log:
```
Motor de IA: Qwen 0.5b (via Hugging Face API)
```

---

## 🌐 Deploy em Railway (Recomendado)

Para rodar 24/7 sem seu PC:

```bash
# 1. Fazer commit e push
git add .
git commit -m "Add Qwen support"
git push origin main

# 2. Importar em Railway.app
# Vá para https://railway.app/ e conecte seu repo
```

Veja instruções completas em: [DEPLOYMENT_RAILWAY.md](./DEPLOYMENT_RAILWAY.md)

---

## 🔧 Configuração Qwen

Em `brain_qwen.py`, você pode ajustar:

```python
QWEN_TEMPERATURE = 0.7      # Criatividade (0.1-1.0)
QWEN_MAX_TOKENS = 150       # Comprimento da resposta
QWEN_TOP_P = 0.9            # Diversidade
QWEN_RETRY_ATTEMPTS = 3     # Tentativas de reconexão
QWEN_RETRY_DELAY = 2        # Espera entre tentativas (seg)
```

---

## 💾 Memória

O bot continua aprendendo:

- **Arquivo JSON**: `data/memoria_dialogos.json`
- **Histórico de threads**: Em memória RAM (até 5 trocas por thread)
- **Limite**: Configurable via `MAX_MEMORIA` em `config.py`

---

## ⚠️ Limitações Conhecidas

| Limitação | Solução |
|-----------|---------|
| Rate limit HF (5 req/min grátis) | Aguarde 1-2 min entre respostas |
| Timeout API (30s) | Implementado retry automático |
| Sem contexto de threads longas | Mantém últimas 5 trocas |
| Sem suporte a imagens | Só texto por enquanto |

---

## 🐛 Troubleshooting

### "InvalidEmail: Address local part cannot be empty"
- **Causa**: `BLUESKY_HANDLE` com `@` no início ou vazio
- **Fix**: Use `dimitriai.bsky.social` (sem `@`)

### "HUGGINGFACE_API_KEY not found"
- **Causa**: Chave não configurada no `.env`
- **Fix**: Crie token em https://huggingface.co/settings/tokens

### "Model is loading" (HTTP 503)
- **Causa**: Modelo Qwen carregando na primeira requisição
- **Fix**: Aguarde 30-60s, a API refaz automaticamente

### "Unauthorized" (HTTP 401)
- **Causa**: Chave de API inválida ou expirada
- **Fix**: Gere uma nova chave em https://huggingface.co/settings/tokens

---

## 📊 Monitoramento

O bot loga tudo em:
- **Console**: Tempo real durante execução
- **Arquivo**: `logs/dimitri.log`

Procure por:
- ✅ `[LOGIN] Autenticado como @...`
- ✅ `Motor de IA: Qwen 0.5b`
- ✅ `[QWEN] Gerou: '...'`
- ❌ `[QWEN] Falhou após 3 tentativas`

---

## 🔗 Recursos

| Recurso | Link |
|---------|------|
| Qwen docs | https://huggingface.co/Qwen/Qwen0.5B |
| HF API docs | https://huggingface.co/docs/api-inference |
| Railway docs | https://docs.railway.app/ |
| Bluesky API | https://github.com/MarshalX/atproto |

---

## 📝 Changelog (vs. versão Markov)

```diff
+ Adicionado brain_qwen.py com integração Qwen 0.5b
+ Adicionado Dockerfile para Railway
+ Adicionado suporte para múltiplos brains (seleção automática)
+ Adicionado retry automático e tratamento de timeouts
+ Adicionado DEPLOYMENT_RAILWAY.md
+ Adicionado dimitri_colab.ipynb para testes rápidos
- Removido hardcoding de brain (agora dinâmico)

✅ Brain Markov ainda funciona se não houver chave HF
```

---

## 🎓 Próximos Passos

1. **Setup local**: Siga Quick Start acima
2. **Teste no Colab**: Use `dimitri_colab.ipynb`
3. **Deploy Railway**: Siga `DEPLOYMENT_RAILWAY.md`
4. **Ajustar parâmetros**: Veja `brain_qwen.py` linhas 55-61

---

**Status**: ✅ Pronto para produção  
**Última atualização**: 2026-06-16  
**Autor**: Dimitri AI Team
