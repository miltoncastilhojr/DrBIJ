# Ajustes Técnicos - Análise Forense de Áudio

## 🔍 1. Picos e Silêncios Ilusórios (>60k)

### **Problema Identificado:**
- Picos acima de 60k são **artifacts** ou **ruído digital**
- Detecção inadequada está capturando **falsos positivos**

### **Causas Técnicas:**
```python
# PROBLEMA: Parâmetros muito sensíveis
peaks, _ = find_peaks(amp, height=np.percentile(amp, 90))  # 90% é muito baixo
silences = np.where(amp < np.percentile(amp, 2))[0]       # 2% é muito baixo
```

### **Soluções Otimizadas:**

#### A) **Detecção Inteligente de Picos:**
```python
def detectar_picos_inteligente(samples, sr):
    """Detecção refinada de picos com filtros"""
    amp = np.abs(samples)
    
    # 1. Filtro de amplitude mínima
    threshold_min = np.std(amp) * 2  # 2 desvios padrão
    
    # 2. Filtro de percentil mais rigoroso
    threshold_high = np.percentile(amp, 95)  # 95% em vez de 90%
    
    # 3. Distância mínima entre picos (evita clusters)
    min_distance = int(sr * 0.1)  # 100ms mínimo entre picos
    
    # 4. Detecção com múltiplos critérios
    peaks, properties = find_peaks(
        amp, 
        height=max(threshold_min, threshold_high),
        distance=min_distance,
        prominence=np.std(amp)  # Proeminência mínima
    )
    
    return peaks, properties
```

#### B) **Detecção Inteligente de Silêncios:**
```python
def detectar_silencios_inteligente(samples, sr):
    """Detecção refinada de silêncios"""
    amp = np.abs(samples)
    
    # 1. Threshold dinâmico baseado em estatísticas
    noise_floor = np.percentile(amp, 5)  # 5% como ruído de fundo
    silence_threshold = noise_floor * 3   # 3x o ruído de fundo
    
    # 2. Duração mínima de silêncio (evita artifacts)
    min_silence_duration = int(sr * 0.5)  # 500ms mínimo
    
    # 3. Detectar zonas silenciosas
    silence_mask = amp < silence_threshold
    
    # 4. Filtrar silêncios muito curtos
    silence_regions = []
    start = None
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            duration = i - start
            if duration >= min_silence_duration:
                silence_regions.append((start, i, duration))
            start = None
    
    return silence_regions
```

---

## 📝 2. Transcrição Sem Timestamp no Relatório

### **Razões Técnicas Para Incluir Timestamps:**

#### A) **Valor Forense:**
- ✅ **Sincronização temporal** com eventos
- ✅ **Verificação de autenticidade** 
- ✅ **Correlação com outras evidências**
- ✅ **Análise de padrões temporais**

#### B) **Análise Técnica:**
- ✅ **Velocidade de fala** (palavras/minuto)
- ✅ **Pausas significativas** 
- ✅ **Mudanças de tom/ritmo**
- ✅ **Identificação de edições**

### **Implementação Melhorada:**
```python
def gerar_transcricao_com_analise_temporal(result):
    """Transcrição com análise temporal detalhada"""
    
    transcricao_completa = []
    estatisticas_tempo = {
        'pausas_longas': [],
        'velocidade_fala': [],
        'duracao_total': 0
    }
    
    for i, segment in enumerate(result["segments"]):
        inicio = segment["start"]
        fim = segment["end"]
        texto = segment["text"].strip()
        duracao = fim - inicio
        
        # Análise de velocidade (palavras por minuto)
        palavras = len(texto.split())
        wpm = (palavras / duracao) * 60 if duracao > 0 else 0
        
        # Detectar pausas longas (>2s entre segmentos)
        if i > 0:
            pausa = inicio - result["segments"][i-1]["end"]
            if pausa > 2.0:
                estatisticas_tempo['pausas_longas'].append({
                    'tempo': inicio,
                    'duracao': pausa
                })
        
        transcricao_completa.append({
            'inicio': formatar_tempo(inicio),
            'fim': formatar_tempo(fim),
            'duracao': f"{duracao:.2f}s",
            'texto': texto,
            'palavras': palavras,
            'velocidade_wpm': f"{wpm:.1f}",
            'timestamp_raw': inicio
        })
        
        estatisticas_tempo['velocidade_fala'].append(wpm)
    
    return transcricao_completa, estatisticas_tempo
```

---

## 📊 3. Espectrograma "Terrível"

### **Problemas Identificados:**
- ❌ **Resolução baixa** (512 nperseg)
- ❌ **Escala inadequada** 
- ❌ **Colormap ruim**
- ❌ **Sem normalização**

### **Espectrograma de Qualidade Forense:**
```python
def gerar_espectrograma_profissional(samples, sr, arquivo):
    """Espectrograma otimizado para análise forense"""
    
    # Parâmetros otimizados
    nperseg = min(2048, len(samples) // 8)  # Melhor resolução
    overlap = nperseg // 2                   # 50% overlap
    
    # Gerar espectrograma com alta resolução
    f, t, Sxx = spectrogram(
        samples, 
        fs=sr, 
        nperseg=nperseg,
        noverlap=overlap,
        window='hann'  # Janela Hann para melhor resolução
    )
    
    # Converter para dB com normalização
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Criar figura com múltiplas visualizações
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Espectrograma principal
    ax1 = plt.subplot(3, 1, 1)
    mappable = ax1.pcolormesh(
        t, f, Sxx_db, 
        shading='gouraud', 
        cmap='plasma',  # Colormap melhor que viridis
        vmin=np.percentile(Sxx_db, 10),  # Contraste dinâmico
        vmax=np.percentile(Sxx_db, 95)
    )
    plt.colorbar(mappable, ax=ax1, label="Intensidade [dB]")
    ax1.set_ylabel("Frequência [Hz]")
    ax1.set_title(f"Espectrograma Forense - {arquivo}")
    ax1.set_ylim(0, sr//2)  # Limitar até Nyquist
    
    # 2. Zoom em frequências de voz (80-8000 Hz)
    ax2 = plt.subplot(3, 1, 2)
    voice_mask = (f >= 80) & (f <= 8000)
    mappable2 = ax2.pcolormesh(
        t, f[voice_mask], Sxx_db[voice_mask], 
        shading='gouraud', 
        cmap='plasma'
    )
    plt.colorbar(mappable2, ax=ax2, label="Intensidade [dB]")
    ax2.set_ylabel("Freq. Voz [Hz]")
    ax2.set_title("Zoom - Frequências de Voz (80-8000 Hz)")
    
    # 3. Análise de energia temporal
    ax3 = plt.subplot(3, 1, 3)
    energia_temporal = np.sum(Sxx, axis=0)
    ax3.plot(t, 10 * np.log10(energia_temporal + 1e-10), 'b-', linewidth=1)
    ax3.set_xlabel("Tempo [s]")
    ax3.set_ylabel("Energia Total [dB]")
    ax3.set_title("Envelope de Energia Temporal")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar em alta qualidade
    plt.savefig(
        os.path.join(PASTA_OUT, f"{arquivo}_espectrograma_hd.png"), 
        dpi=300, 
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()
    
    return f, t, Sxx_db
```

---

## 🎬 4. Extração de Áudio de Vídeo

### **Complexidade: BAIXA** ⭐⭐

### **Implementação Simples:**
```python
def extrair_audio_de_video(video_path, output_path):
    """Extrai áudio de arquivo de vídeo"""
    try:
        command = [
            'ffmpeg', '-y',
            '-i', video_path,           # Input vídeo
            '-vn',                      # Sem vídeo
            '-acodec', 'pcm_s16le',     # Codec áudio
            '-ar', '16000',             # Sample rate
            '-ac', '1',                 # Mono
            '-af', 'volume=1.5',        # Boost volume se necessário
            output_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao extrair áudio: {e.stderr}")
        return False

# Adicionar ao processamento principal:
def processar_video_e_audio(path, pbar):
    """Processa vídeo extraindo áudio primeiro"""
    base = os.path.splitext(os.path.basename(path))[0]
    
    # Verificar se é vídeo
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
    if os.path.splitext(path)[1].lower() in video_extensions:
        
        # Extrair áudio do vídeo
        audio_temp = os.path.join(PASTA_OUT, f"{base}_extracted.wav")
        if extrair_audio_de_video(path, audio_temp):
            # Processar o áudio extraído
            return processar_audio_simples(audio_temp, pbar)
    
    # Se não é vídeo, processar como áudio normal
    return processar_audio_simples(path, pbar)
```

### **Formatos Suportados:**
- ✅ MP4, AVI, MOV, MKV
- ✅ WMV, FLV, M4V, WEBM
- ✅ Qualquer formato que o FFmpeg suporte

---

## 🤖 5. Por que o Copilot é "Maluco"?

### **Diferenças Técnicas:**

| Aspecto | Claude (Eu) | Copilot |
|---------|-------------|---------|
| **Foco** | Conversação + Análise | Completar código |
| **Contexto** | Toda conversa | Arquivo atual |
| **Objetivo** | Entender problema | Sugerir sintaxe |
| **Análise** | Arquitetura completa | Linha por linha |
| **Debugging** | Causa raiz | Fix rápido |

### **Por que Parece "Maluco":**

#### A) **Contexto Limitado:**
- 🤖 Copilot: "Vê" só o arquivo atual
- 🧠 Claude: Entende o projeto completo

#### B) **Objetivo Diferente:**
- 🤖 Copilot: "Complete esta linha"
- 🧠 Claude: "Resolva este problema"

#### C) **Treinamento:**
- 🤖 Copilot: Baseado em GitHub (padrões)
- 🧠 Claude: Baseado em conversações (lógica)

### **Quando Usar Cada Um:**
- **Copilot**: ✅ Autocompletar, snippets, boilerplate
- **Claude**: ✅ Arquitetura, debugging, análise técnica

---

## 🛠️ Implementação dos Ajustes

Quer que eu implemente essas melhorias agora? Posso criar:

1. **Script com detecção inteligente** de picos/silêncios
2. **Relatórios com análise temporal** completa  
3. **Espectrogramas HD** profissionais
4. **Suporte a vídeo** automático
5. **Configuração flexível** para todos os parâmetros

**Qual ajuste implemento primeiro?** 🎯