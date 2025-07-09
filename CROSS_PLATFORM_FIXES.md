# Cross-Platform Compatibility Fixes

## Problema Original
"O script está apresentando problemas em diferentes ambientes que não são o meu"

## Problemas Identificados e Soluções

### 1. **Dependências Faltando ou Incompatíveis**

**Problema**: Script crashava imediatamente quando dependências estavam ausentes
**Solução**: 
- Imports envolvidos em try/catch com mensagens de erro claras
- Instruções específicas de instalação para cada dependência
- Verificação de FFmpeg como dependência externa crítica

```python
try:
    import numpy as np
except ImportError as e:
    print(f"❌ Erro: numpy não encontrado. Instale com: pip install numpy")
    sys.exit(1)
```

### 2. **FFmpeg Não Detectado**

**Problema**: Script assumia que FFmpeg estava sempre disponível
**Solução**:
- Função `verificar_ffmpeg()` que detecta disponibilidade
- Suporte a diferentes nomes (ffmpeg vs ffmpeg.exe no Windows)
- Instruções de instalação específicas por plataforma

```python
def verificar_ffmpeg():
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        return False, "FFmpeg não encontrado no PATH do sistema"
    # Testa se funciona
    result = subprocess.run([ffmpeg_path, '-version'], ...)
```

### 3. **Problemas de Encoding**

**Problema**: Falhas em sistemas com diferentes locales
**Solução**:
- Tentativas múltiplas de encoding (utf-8, utf-8-sig, latin1, cp1252)
- Especificação explícita de encoding em todas operações de arquivo
- Fallback gracioso quando encoding falha

```python
encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
for encoding in encodings_to_try:
    try:
        with open(arquivo, "r", encoding=encoding) as f:
            data = json.load(f)
        break
    except UnicodeDecodeError:
        continue
```

### 4. **Whisper Opcional**

**Problema**: Script falhava completamente sem Whisper
**Solução**:
- Whisper tornado opcional com fallback
- Módulo dummy quando Whisper não disponível
- Transcrição continua indicando indisponibilidade

```python
WHISPER_AVAILABLE = True
try:
    import whisper
except ImportError:
    WHISPER_AVAILABLE = False
    class DummyWhisper:
        @staticmethod
        def transcribe(*args, **kwargs):
            return {"text": "[Transcrição indisponível - Whisper não instalado]"}
    whisper = DummyWhisper()
```

### 5. **Detecção de Plataforma**

**Problema**: Operações específicas de sistema não funcionavam cross-platform
**Solução**:
- Função `verificar_plataforma()` detecta sistema operacional
- Lógica específica para Windows, Linux e macOS
- Comandos e caminhos adaptados por plataforma

```python
def verificar_plataforma():
    sistema = platform.system().lower()
    return {
        'sistema': sistema,
        'is_windows': sistema == 'windows',
        'is_linux': sistema == 'linux',
        'is_macos': sistema == 'darwin'
    }
```

### 6. **Tratamento de Erros Melhorado**

**Problema**: Mensagens de erro pouco úteis
**Solução**:
- Mensagens específicas por tipo de erro
- Dicas de solução de problemas por plataforma
- Verificação de permissões e recursos

```python
except Exception as e:
    logger.error(f"❌ Erro crítico: {e}")
    if plataforma_info.get('is_windows'):
        logger.error("   - Execute como administrador se necessário")
    elif plataforma_info.get('is_linux'):
        logger.error("   - Verifique permissões: chmod +x script.py")
```

## Melhorias de Compatibilidade

### Sistemas Suportados
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Debian, CentOS, RHEL)
- ✅ macOS (Intel e Apple Silicon)

### Versões Python
- ✅ Python 3.8+
- ✅ Python 3.9+
- ✅ Python 3.10+
- ✅ Python 3.11+
- ✅ Python 3.12+

### Dependências Opcionais
- ✅ Script funciona sem Whisper (transcrição limitada)
- ✅ Matplotlib usa backend não-interativo
- ✅ FFmpeg detectado automaticamente

## Instruções de Instalação por Plataforma

### Windows
```bash
# Instalar Python dependencies
pip install -r requirements.txt

# Instalar FFmpeg
winget install ffmpeg
# OU baixar de https://ffmpeg.org/download.html
```

### Linux (Ubuntu/Debian)
```bash
# Instalar dependências do sistema
sudo apt-get update
sudo apt-get install ffmpeg python3-pip

# Instalar Python dependencies
pip3 install -r requirements.txt
```

### Linux (CentOS/RHEL)
```bash
# Instalar dependências do sistema
sudo yum install ffmpeg python3-pip
# OU: sudo dnf install ffmpeg python3-pip

# Instalar Python dependencies
pip3 install -r requirements.txt
```

### macOS
```bash
# Instalar FFmpeg com Homebrew
brew install ffmpeg

# Instalar Python dependencies
pip3 install -r requirements.txt
```

## Teste de Compatibilidade

Execute o script de teste incluído:
```bash
python3 test_environment_fixes.py
```

Este teste verifica:
- ✅ Compilação do script
- ✅ Detecção de dependências faltando
- ✅ Funções de detecção de plataforma
- ✅ Tratamento de erros melhorado

## Solução de Problemas

### FFmpeg não encontrado
```
❌ FFmpeg não disponível: FFmpeg não encontrado no PATH do sistema
   Baixe FFmpeg de: https://ffmpeg.org/download.html
   Ou instale com: winget install ffmpeg
```

### Permissões negadas
```
❌ Erro de permissão no diretório de saída
   Dica: Verifique permissões de arquivo e diretório
```

### Dependências Python faltando
```
❌ Erro: numpy não encontrado. Instale com: pip install numpy
❌ Erro: scipy não encontrado. Instale com: pip install scipy
```

### Whisper indisponível
```
⚠️  Aviso: Whisper não encontrado. Transcrição será desabilitada.
   Instale com: pip install openai-whisper
```

Estas melhorias resolvem o problema principal de compatibilidade entre diferentes ambientes, fornecendo feedback claro e soluções acionáveis quando problemas ocorrem.