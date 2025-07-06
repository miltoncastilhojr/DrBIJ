# Como Desativar a Análise Emocional

## Opção 1: Use o Script Simplificado (RECOMENDADO)

### 📁 Use o arquivo: `audio_forensic_no_emotions.py`

Este script já vem com a análise emocional **COMPLETAMENTE DESATIVADA**:

```bash
python audio_forensic_no_emotions.py
```

**O que foi removido:**
- ✅ Categorização emocional LIWC
- ✅ Gráficos de padrões emocionais
- ✅ Análise de sentimentos
- ✅ Base de dados de texto emocional
- ✅ Referências a arquivos `liwc_categorias.json` e `mensagens_ge_london_categorizadas.csv`

**O que permanece:**
- ✅ Transcrição de áudio (Whisper)
- ✅ Análise técnica (MFCC, espectrograma)
- ✅ Detecção de picos e silêncios
- ✅ Geração de SRT
- ✅ Relatórios PDF técnicos
- ✅ Hash de integridade

---

## Opção 2: Modificar o Script Original

Se você quiser modificar seu script original, faça estas alterações:

### 1. Remover Carregamento de Categorias LIWC

**REMOVER estas linhas:**
```python
# === CARREGAR CATEGORIAS LIWC ===
with open(CATEGORIAS_JSON, "r", encoding="utf-8") as f:
    LIWC_CATEGORIAS = json.load(f)
```

**SUBSTITUIR por:**
```python
# === ANÁLISE EMOCIONAL DESATIVADA ===
LIWC_CATEGORIAS = {}  # Vazio - sem categorias
```

### 2. Simplificar a Função de Categorização

**MODIFICAR a função `categorizar_audio`:**
```python
def categorizar_audio(texto):
    # Retorna sempre "tecnico" em vez de analisar emoções
    return "tecnico"
```

### 3. Remover Base de Texto Emocional

**REMOVER estas linhas:**
```python
# === CARREGAR BASE DE TEXTO CATEGORIZADO ===
if os.path.exists(TEXTO_CSV):
    texto_df = pd.read_csv(TEXTO_CSV)
else:
    texto_df = pd.DataFrame(columns=["timestamp", "mensagem", "categoria"])
```

**SUBSTITUIR por:**
```python
# === BASE DE TEXTO DESATIVADA ===
texto_df = pd.DataFrame()  # DataFrame vazio
```

### 4. Desativar Gráficos Emocionais

**MODIFICAR a função `identificar_padroes_textuais`:**
```python
def identificar_padroes_textuais():
    # Não gera gráficos emocionais
    return {}
```

### 5. Simplificar Geração de PDF

**MODIFICAR a função `gerar_pdf`:**
```python
def gerar_pdf(arquivo, duracao, hash_val, picos, sils, fingerprint, transcricao, categoria):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Relatorio Tecnico - {arquivo}", ln=True, align='L')
    
    # Remover referências a padrões emocionais
    pdf.multi_cell(0, 10, f"Duracao: {duracao:.2f} segundos\nHash SHA256: {hash_val}\nPicos detectados: {picos}\nZonas de silencio: {sils}\nFingerprint MFCC: {fingerprint}\n\nTranscricao:\n{transcricao}")

    espectro_path = os.path.join(PASTA_OUT, f"{arquivo}_espectrograma.png")
    if os.path.exists(espectro_path):
        pdf.image(espectro_path, w=180)

    pdf.output(os.path.join(PASTA_OUT, f"{arquivo}_relatorio_tecnico.pdf"))
```

---

## Opção 3: Configuração Rápida

### Adicione uma configuração no topo do script:

```python
# === CONFIGURAÇÃO RÁPIDA ===
ANALISE_EMOCIONAL_ATIVADA = False  # Mude para True se quiser ativar

# Use esta configuração nas funções:
def categorizar_audio(texto):
    if not ANALISE_EMOCIONAL_ATIVADA:
        return "tecnico"
    
    # Código original da análise emocional aqui...
    texto_lower = texto.lower()
    for categoria, palavras in LIWC_CATEGORIAS.items():
        if any(p in texto_lower for p in palavras):
            return categoria
    return "outro"

def identificar_padroes_textuais():
    if not ANALISE_EMOCIONAL_ATIVADA:
        return {}
    
    # Código original aqui...
```

---

## Comparação dos Métodos

| Método | Dificuldade | Tempo | Recomendação |
|--------|-------------|-------|--------------|
| **Script Novo** | ⭐ Fácil | 2 min | ✅ **MELHOR** |
| **Modificar Original** | ⭐⭐⭐ Médio | 15 min | ⚠️ Se necessário |
| **Configuração Rápida** | ⭐⭐ Fácil | 5 min | ✅ Opção rápida |

---

## Vantagens da Versão Sem Emoções

### 🚀 **Performance**
- **Mais rápido**: Sem processamento emocional
- **Menos memória**: Sem carregar bases de dados extras
- **Menos dependências**: Não precisa dos arquivos JSON/CSV

### 🔧 **Simplicidade**
- **Foco técnico**: Apenas análise forense
- **Menos erros**: Sem dependências externas
- **Mais estável**: Menos pontos de falha

### 📊 **Resultados**
- **Relatórios limpos**: Sem informações emocionais desnecessárias
- **Dados técnicos**: MFCC, espectrogramas, transcrições
- **Forense puro**: Foco na evidência técnica

---

## Teste da Configuração

### Para verificar se está funcionando:

1. **Execute o script**
2. **Verifique os logs**: Deve aparecer "MODO: Análise técnica"
3. **Verifique os PDFs**: Devem ser "Relatório Técnico" (não "Relatório Forense")
4. **Verifique o CSV**: Coluna "categoria" deve ter sempre "tecnico"

### Exemplo de saída esperada:
```
🚀 Iniciando análise forense técnica (SEM análise emocional)
🔧 MODO: Análise técnica (análise emocional DESATIVADA)
✅ Processamento técnico concluído! 3 arquivos processados
ℹ️  Análise emocional foi DESATIVADA - apenas dados técnicos extraídos
```

---

## Dúvidas Frequentes

### ❓ **"Posso reativar as emoções depois?"**
✅ **Sim!** Basta mudar `enable_emotions: True` no CONFIG ou usar o script original.

### ❓ **"A transcrição ainda funciona?"**
✅ **Sim!** O Whisper continua funcionando normalmente, apenas sem análise emocional.

### ❓ **"Os espectrogramas ainda são gerados?"**
✅ **Sim!** Toda análise técnica permanece ativa.

### ❓ **"Preciso dos arquivos JSON e CSV?"**
❌ **Não!** A versão sem emoções não precisa desses arquivos.

---

## Recomendação Final

### 🎯 **USE**: `audio_forensic_no_emotions.py`

**Motivos:**
- ✅ Já otimizado e testado
- ✅ Sem riscos de erro
- ✅ Mantém todas as funcionalidades técnicas
- ✅ Performance melhor
- ✅ Código mais limpo

**Para usar:**
```bash
# Coloque os arquivos na pasta INA/
python audio_forensic_no_emotions.py
```

**Resultados em:** `OUTA/`