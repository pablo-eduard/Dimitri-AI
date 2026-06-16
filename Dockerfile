# Usa Python 3.11 slim como base (menor imagem)
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Instala dependências com cache busting mínimo
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código
COPY . .

# Garante que os diretórios existem
RUN mkdir -p data logs

# Comando para rodar o bot
CMD ["python", "main.py"]
