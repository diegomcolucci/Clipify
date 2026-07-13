# Clipify — Guia de Execução

Referência rápida para rodar o projeto Clipify (UI Gradio) em qualquer máquina.

---

## ✅ Pré-requisitos

- Python 3.11
- `git`
- `ffmpeg`
- Chave de API (Google Gemini, OpenAI ou OpenRouter)

### Instalar ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```powershell
# Via Chocolatey
choco install ffmpeg

# Ou via winget
winget install FFmpeg.Foundation.FFmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

---

## 🚀 Primeira vez no computador

### 1. Clonar o repositório
```bash
cd ~/Projects
git clone https://github.com/diegomcolucci/Clipify.git
cd Clipify
```

### 2. Criar ambiente virtual e instalar dependências

**macOS/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurar chaves de API
```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione sua chave:

```env
# Google Gemini (recomendado - gratuito)
GOOGLE_API_KEY=sua_chave_aqui
AI_PROVIDER=gemini

# OU OpenAI
OPENAI_API_KEY=sua_chave_aqui
AI_PROVIDER=openai

# OU OpenRouter (múltiplos modelos, incluindo gratuitos)
OPENROUTER_API_KEY=sua_chave_aqui
OPENROUTER_MODEL=auto
AI_PROVIDER=openrouter
```

**Obter chaves:**
- Google AI Studio: https://aistudio.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys
- OpenRouter: https://openrouter.ai/settings/keys

---

## ▶️ Rodar a UI

**macOS/Linux:**
```bash
source venv/bin/activate
python ui/app_gradio.py
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
python ui/app_gradio.py
```

Acesse no navegador:
```
http://127.0.0.1:7860
```

---

## 🧪 Verificar se está tudo funcionando

### Verificação rápida de sintaxe
```bash
python -m py_compile ui/app_gradio.py clipify/pipelines/ui_helpers.py clipify/video/processor.py clipify/pipelines/gemini_pipeline.py
```

### Rodar testes unitários
```bash
python -m pytest tests/test_providers.py -v
```

### Teste completo
1. Rode o app com `python ui/app_gradio.py`
2. Faça upload de um vídeo curto
3. Clique em **Auto-suggest params** (opcional)
4. Clique em **Generate**
5. Verifique se os clips gerados têm áudio e legenda corretos

---

## 🛠️ Comandos úteis

### Ver status do Git
```bash
git status --short
git log --oneline -5
```

### Atualizar o projeto
```bash
git pull origin develop
```

### Reinstalar dependências
```bash
pip install -r requirements.txt --force-reinstall
```

---

## ⚠️ Problemas comuns

### Porta 7860 já está em uso
```bash
# Encontrar o processo
netstat -ano | Select-String ":7860"

# Encerrar (substitua 12345 pelo PID)
# Windows:
taskkill /F /PID 12345

# macOS/Linux:
kill -9 12345
```

### Erro de API key
Verifique se o arquivo `.env` existe e tem a chave correta:
```bash
cat .env
```

### Transcrição lenta na primeira vez
O Whisper baixa o modelo `base` na primeira execução. Isso é normal.

---

## 📁 Arquivos importantes

- `ui/app_gradio.py` — interface Gradio
- `clipify/pipelines/gemini_pipeline.py` — highlights e alinhamento
- `clipify/pipelines/ui_helpers.py` — ffmpeg + transcrição Whisper
- `clipify/video/processor.py` — queima de legendas
- `.env` — suas chaves de API (NÃO commitar)
