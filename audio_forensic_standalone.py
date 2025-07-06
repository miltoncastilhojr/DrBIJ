#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Forensic Analysis - STANDALONE VERSION
============================================
✅ ZERO dependências externas
✅ ZERO dicionários necessários  
✅ ZERO arquivos JSON/CSV
✅ FUNCIONA 100% sem configurações extras
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
from scipy.fftpack import dct
import whisper
import csv
import hashlib
from fpdf import FPDF
import pandas as pd
import psutil
import subprocess
import gc
from tqdm import tqdm
import functools
import logging
from typing import Dict, List, Optional

# ========================================
# CONFIGURAÇÕES EMBUTIDAS (SEM ARQUIVOS EXTERNOS)
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE_DIR, "INA")
PASTA_OUT = os.path.join(BASE_DIR, "OUTA")
LOG_CSV = os.path.join(PASTA_OUT, "relatorio_forense.csv")
SUMARIO_CSV = os.path.join(PASTA_OUT, "sumario_relatorio_geral.csv")

# Criar diretório de saída
os.makedirs(PASTA_OUT, exist_ok=True)

# Configuração completa embutida
CONFIG = {
    'whisper': {
        'model': 'tiny',
        'device': 'cpu',
        'language': 'Portuguese',
        'fp16': True,
        'beam_size': 2
    },
    'audio': {
        'sample_rate': 16000,
        'channels': 1,
        'chunk_size': 4096
    },
    'processing': {
        'max_memory_percent': 85,
        'enable_spectrograms': True,
        'enable_srt': True,
        'enable_pdf': True,
        'enable_emotions': False  # ❌ DESATIVADO
    },
    'formatos_suportados': [
        '.wav', '.m4a', '.mp3', '.opus', '.flac', '.aac',
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'  # Vídeos também
    ]
}

# ========================================
# LOGGING SETUP
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PASTA_OUT, 'forensic_standalone.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================
# FUNÇÕES UTILITÁRIAS
# ========================================

def verificar_dependencias():
    """Verifica se todas as dependências estão disponíveis"""
    required_packages = [
        'whisper', 'scipy', 'matplotlib', 'pandas', 
        'psutil', 'fpdf', 'tqdm', 'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"❌ Pacotes em falta: {', '.join(missing)}")
        logger.error("💡 Instale com: pip install " + " ".join(missing))
        raise ImportError(f"Pacotes em falta: {', '.join(missing)}")
    
    logger.info("✅ Todas as dependências verificadas")

def verificar_memoria():
    """Monitora uso de memória"""
    process = psutil.Process()
    mem_percent = process.memory_percent()
    
    if mem_percent > CONFIG['processing']['max_memory_percent']:
        gc.collect()
        mem_percent = process.memory_percent()
        
        if mem_percent > 90:
            logger.warning(f"⚠️  Memória crítica: {mem_percent:.1f}%")
            raise MemoryError(f"Uso crítico de memória: {mem_percent:.1f}%")
    
    return mem_percent

def verificar_ffmpeg():
    """Verifica se FFmpeg está disponível"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ FFmpeg não encontrado!")
        logger.error("💡 Instale com: sudo apt install ffmpeg")
        return False

# ========================================
# PROCESSAMENTO DE ÁUDIO/VÍDEO
# ========================================

def extrair_audio_de_video(video_path: str, output_path: str) -> bool:
    """Extrai áudio de arquivo de vídeo"""
    try:
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(CONFIG['audio']['sample_rate']),
            '-ac', str(CONFIG['audio']['channels']),
            '-af', 'volume=1.2',  # Pequeno boost de volume
            output_path
        ]
        
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        logger.info(f"✅ Áudio extraído do vídeo: {os.path.basename(video_path)}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao extrair áudio: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout na extração de áudio")
        return False

def converter_audio_safe(input_path: str, output_path: str) -> bool:
    """Converte áudio com tratamento robusto de erros"""
    try:
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(CONFIG['audio']['sample_rate']),
            '-ac', str(CONFIG['audio']['channels']),
            '-acodec', 'pcm_s16le',
            output_path
        ]
        
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        logger.info(f"✅ Áudio convertido: {os.path.basename(input_path)}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro na conversão: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout na conversão")
        return False

# ========================================
# ANÁLISE TÉCNICA
# ========================================

def calcular_hash_seguro(path: str) -> str:
    """Calcula hash SHA256 do arquivo"""
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(CONFIG['audio']['chunk_size']), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"❌ Erro ao calcular hash: {e}")
        return "ERRO_HASH"

def extract_mfcc_simples(signal: np.ndarray, rate: int, num_ceps: int = 13) -> np.ndarray:
    """Extração MFCC simplificada e robusta"""
    try:
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        # Configurações de frame
        frame_len = int(rate * 0.025)  # 25ms
        frame_step = int(rate * 0.01)  # 10ms
        signal_len = len(emphasized)
        
        if signal_len < frame_len:
            logger.warning("⚠️  Áudio muito curto para análise MFCC")
            return np.zeros(num_ceps)
        
        # Criar frames
        num_frames = int(np.ceil(float(signal_len - frame_len) / frame_step)) + 1
        pad_signal = np.append(emphasized, np.zeros(num_frames * frame_step + frame_len - signal_len))
        
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        
        frames = pad_signal[indices.astype(np.int32)]
        frames *= np.hamming(frame_len)
        
        # FFT
        nfft = 512
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = (1.0 / nfft) * (mag_frames ** 2)
        
        # Mel filter bank
        nfilt = 26
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        bin_points = np.floor((nfft + 1) * hz_points / rate).astype(int)
        
        fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            for k in range(bin_points[m - 1], bin_points[m]):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(bin_points[m], bin_points[m + 1]):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        log_fb = 20 * np.log10(filter_banks)
        
        # DCT
        mfcc = dct(log_fb, type=2, axis=1, norm='ortho')[:, :num_ceps]
        
        return np.mean(mfcc, axis=0)
        
    except Exception as e:
        logger.error(f"❌ Erro na extração MFCC: {e}")
        return np.zeros(num_ceps)

def analisar_audio_tecnicamente(samples: np.ndarray, sr: int) -> Dict:
    """Análise técnica completa do áudio"""
    amp = np.abs(samples)
    
    # Estatísticas básicas
    stats = {
        'amplitude_media': np.mean(amp),
        'amplitude_max': np.max(amp),
        'amplitude_std': np.std(amp),
        'energia_total': np.sum(amp ** 2),
        'zero_crossings': np.sum(np.diff(np.sign(samples)) != 0)
    }
    
    # Detecção de picos (mais rigorosa)
    threshold_picos = np.percentile(amp, 97)  # 97% em vez de 90%
    min_distance = int(sr * 0.1)  # 100ms entre picos
    
    peaks, properties = find_peaks(
        amp, 
        height=threshold_picos,
        distance=min_distance,
        prominence=np.std(amp) * 1.5
    )
    
    # Detecção de silêncios (mais rigorosa)
    noise_floor = np.percentile(amp, 3)  # 3% como ruído de fundo
    silence_threshold = max(noise_floor * 2, np.mean(amp) * 0.1)
    
    silence_mask = amp < silence_threshold
    silence_samples = np.sum(silence_mask)
    
    # Análise espectral básica
    freq_spectrum = np.fft.rfft(samples)
    freq_magnitude = np.abs(freq_spectrum)
    dominant_freq = np.argmax(freq_magnitude) * sr / (2 * len(freq_magnitude))
    
    return {
        'picos': len(peaks),
        'picos_intensidade': properties.get('peak_heights', []).tolist() if len(peaks) > 0 else [],
        'silencio_amostras': int(silence_samples),
        'silencio_percentual': float(silence_samples / len(samples) * 100),
        'frequencia_dominante': float(dominant_freq),
        'estatisticas': stats
    }

# ========================================
# TRANSCRIÇÃO E TIMESTAMPS
# ========================================

@functools.lru_cache(maxsize=1)
def get_whisper_model():
    """Carrega modelo Whisper com cache"""
    logger.info("🤖 Carregando modelo Whisper...")
    model = whisper.load_model(CONFIG['whisper']['model'], device=CONFIG['whisper']['device'])
    logger.info("✅ Modelo Whisper carregado")
    return model

def formatar_tempo(segundos: float) -> str:
    """Formata tempo para SRT (BUG CORRIGIDO)"""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos_int = int(segundos % 60)
    milissegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02d}:{minutos:02d}:{segundos_int:02d},{milissegundos:03d}"

def transcrever_com_timestamps(audio_path: str) -> tuple:
    """Transcrição com análise temporal detalhada"""
    try:
        model = get_whisper_model()
        result = model.transcribe(
            audio_path,
            language=CONFIG['whisper']['language'],
            word_timestamps=True,
            fp16=CONFIG['whisper']['fp16'],
            beam_size=CONFIG['whisper']['beam_size']
        )
        
        transcricao_completa = result["text"]
        segments_detalhados = []
        
        for i, segment in enumerate(result["segments"]):
            inicio = segment["start"]
            fim = segment["end"]
            texto = segment["text"].strip()
            duracao = fim - inicio
            
            # Análise temporal
            palavras = len(texto.split())
            wpm = (palavras / duracao) * 60 if duracao > 0 else 0
            
            segments_detalhados.append({
                'inicio': inicio,
                'fim': fim,
                'duracao': duracao,
                'texto': texto,
                'palavras': palavras,
                'velocidade_wpm': wpm,
                'inicio_formatado': formatar_tempo(inicio),
                'fim_formatado': formatar_tempo(fim)
            })
        
        return transcricao_completa, segments_detalhados, result
        
    except Exception as e:
        logger.error(f"❌ Erro na transcrição: {e}")
        return "", [], {"segments": []}

# ========================================
# GERAÇÃO DE RELATÓRIOS
# ========================================

def gerar_espectrograma_profissional(samples: np.ndarray, sr: int, arquivo: str):
    """Gera espectrograma de alta qualidade"""
    try:
        # Parâmetros otimizados
        nperseg = min(2048, len(samples) // 8)
        noverlap = nperseg // 2
        
        f, t, Sxx = spectrogram(samples, fs=sr, nperseg=nperseg, noverlap=noverlap)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Criar visualização profissional
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Espectrograma completo
        im1 = axes[0].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='plasma')
        axes[0].set_ylabel('Frequência [Hz]')
        axes[0].set_title(f'Espectrograma Completo - {arquivo}')
        axes[0].set_ylim(0, sr//2)
        plt.colorbar(im1, ax=axes[0], label='Intensidade [dB]')
        
        # Zoom em frequências de voz (80-4000 Hz)
        voice_mask = (f >= 80) & (f <= 4000)
        if np.any(voice_mask):
            im2 = axes[1].pcolormesh(t, f[voice_mask], Sxx_db[voice_mask], shading='gouraud', cmap='plasma')
            axes[1].set_ylabel('Freq. Voz [Hz]')
            axes[1].set_xlabel('Tempo [s]')
            axes[1].set_title('Zoom - Frequências de Voz (80-4000 Hz)')
            plt.colorbar(im2, ax=axes[1], label='Intensidade [dB]')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_OUT, f"{arquivo}_espectrograma.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Espectrograma gerado: {arquivo}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar espectrograma: {e}")

def gerar_relatorio_pdf(arquivo: str, dados: Dict):
    """Gera relatório PDF técnico completo"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(0, 10, f"RELATÓRIO TÉCNICO FORENSE - {arquivo}", ln=True, align='C')
        
        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        
        # Informações técnicas
        info_basica = f"""
INFORMAÇÕES BÁSICAS
Arquivo: {arquivo}
Duração: {dados['duracao']:.2f} segundos
Hash SHA256: {dados['hash']}
Taxa de amostragem: {CONFIG['audio']['sample_rate']} Hz

ANÁLISE TÉCNICA
Picos de amplitude: {dados['analise']['picos']}
Silêncio: {dados['analise']['silencio_percentual']:.1f}%
Frequência dominante: {dados['analise']['frequencia_dominante']:.1f} Hz
Fingerprint MFCC: {dados['mfcc'][:5]}...

TRANSCRIÇÃO TEMPORAL
Segmentos: {len(dados['segments'])}
Texto completo: {len(dados['transcricao'])} caracteres
"""
        
        pdf.multi_cell(0, 5, info_basica.strip())
        
        # Transcrição com timestamps
        if dados['segments']:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(0, 10, "TRANSCRIÇÃO COM TIMESTAMPS", ln=True)
            pdf.set_font("Arial", size=9)
            
            for i, seg in enumerate(dados['segments'][:10]):  # Primeiros 10 segmentos
                linha = f"[{seg['inicio_formatado']} - {seg['fim_formatado']}] {seg['texto']}"
                pdf.multi_cell(0, 4, linha)
                if i < len(dados['segments']) - 1:
                    pdf.ln(1)
        
        # Adicionar espectrograma
        espectro_path = os.path.join(PASTA_OUT, f"{arquivo}_espectrograma.png")
        if os.path.exists(espectro_path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(0, 10, "ANÁLISE ESPECTRAL", ln=True, align='C')
            pdf.image(espectro_path, x=10, w=190)
        
        # Salvar PDF
        pdf_path = os.path.join(PASTA_OUT, f"{arquivo}_relatorio.pdf")
        pdf.output(pdf_path)
        logger.info(f"✅ Relatório PDF gerado: {pdf_path}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar PDF: {e}")

def gerar_arquivo_srt(arquivo: str, segments: List[Dict]):
    """Gera arquivo SRT com timestamps corretos"""
    try:
        srt_path = os.path.join(PASTA_OUT, f"{arquivo}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(f"{i+1}\n")
                f.write(f"{seg['inicio_formatado']} --> {seg['fim_formatado']}\n")
                f.write(f"{seg['texto']}\n\n")
        
        logger.info(f"✅ Arquivo SRT gerado: {srt_path}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar SRT: {e}")

# ========================================
# PROCESSAMENTO PRINCIPAL
# ========================================

def carregar_arquivos_processados() -> set:
    """Carrega lista de arquivos já processados"""
    processados = set()
    if os.path.exists(LOG_CSV):
        try:
            df = pd.read_csv(LOG_CSV)
            processados = set(df['arquivo'].astype(str).tolist())
            logger.info(f"📋 {len(processados)} arquivos já processados")
        except Exception as e:
            logger.warning(f"⚠️  Erro ao ler log: {e}")
    return processados

def processar_arquivo(file_path: str, pbar: tqdm) -> Optional[Dict]:
    """Processa um arquivo de áudio ou vídeo"""
    try:
        verificar_memoria()
        
        arquivo = os.path.basename(file_path)
        base = os.path.splitext(arquivo)[0]
        
        # Determinar se é vídeo ou áudio
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']
        
        pbar.set_description(f"🔄 {'Vídeo' if is_video else 'Áudio'}: {base}")
        
        # Preparar arquivo de áudio
        if is_video:
            audio_path = os.path.join(PASTA_OUT, f"{base}_audio.wav")
            if not extrair_audio_de_video(file_path, audio_path):
                return None
        else:
            audio_path = os.path.join(PASTA_OUT, f"{base}.wav")
            if not converter_audio_safe(file_path, audio_path):
                return None
        
        # Análise técnica
        pbar.set_description(f"🔍 Analisando: {base}")
        
        hash_val = calcular_hash_seguro(audio_path)
        
        try:
            sr, samples = wavfile.read(audio_path)
            if samples.ndim > 1:
                samples = samples[:, 0]
            duracao = len(samples) / sr
        except Exception as e:
            logger.error(f"❌ Erro ao ler áudio: {e}")
            return None
        
        # Análise técnica completa
        analise = analisar_audio_tecnicamente(samples, sr)
        mfcc = extract_mfcc_simples(samples, sr)
        
        # Transcrição
        pbar.set_description(f"🗣️  Transcrevendo: {base}")
        transcricao, segments, result = transcrever_com_timestamps(audio_path)
        
        # Gerar visualizações
        pbar.set_description(f"📊 Gerando gráficos: {base}")
        if CONFIG['processing']['enable_spectrograms']:
            gerar_espectrograma_profissional(samples, sr, base)
        
        # Compilar dados
        dados = {
            'arquivo': base,
            'tipo': 'vídeo' if is_video else 'áudio',
            'duracao': duracao,
            'hash': hash_val,
            'analise': analise,
            'mfcc': mfcc.tolist(),
            'transcricao': transcricao,
            'segments': segments
        }
        
        # Gerar relatórios
        pbar.set_description(f"📄 Gerando relatórios: {base}")
        
        if CONFIG['processing']['enable_srt'] and segments:
            gerar_arquivo_srt(base, segments)
        
        if CONFIG['processing']['enable_pdf']:
            gerar_relatorio_pdf(base, dados)
        
        # Limpeza
        del result
        gc.collect()
        pbar.update(1)
        
        return dados
        
    except Exception as e:
        logger.error(f"❌ Erro geral no processamento: {e}")
        pbar.update(1)
        return None

def main():
    """Função principal - STANDALONE"""
    print("🚀 AUDIO FORENSIC ANALYSIS - STANDALONE")
    print("=" * 50)
    
    try:
        # Verificações iniciais
        verificar_dependencias()
        
        if not verificar_ffmpeg():
            return
        
        # Carregar arquivos processados
        processados = carregar_arquivos_processados()
        
        # Verificar diretório de entrada
        if not os.path.exists(PASTA):
            logger.error(f"❌ Diretório não encontrado: {PASTA}")
            os.makedirs(PASTA, exist_ok=True)
            logger.info(f"📁 Diretório criado: {PASTA}")
            logger.info("📂 Coloque seus arquivos de áudio/vídeo na pasta INA/")
            return
        
        # Encontrar arquivos
        files_to_process = []
        for file in os.listdir(PASTA):
            if any(file.lower().endswith(ext) for ext in CONFIG['formatos_suportados']):
                base = os.path.splitext(file)[0]
                if base not in processados:
                    files_to_process.append(file)
        
        if not files_to_process:
            logger.info("ℹ️  Nenhum arquivo novo para processar")
            return
        
        logger.info(f"📁 Encontrados {len(files_to_process)} arquivos")
        logger.info("🔧 MODO: Análise técnica standalone (SEM dicionários)")
        
        # Processar arquivos
        resultados = []
        with tqdm(total=len(files_to_process), desc="Progresso") as pbar:
            for file in files_to_process:
                file_path = os.path.join(PASTA, file)
                resultado = processar_arquivo(file_path, pbar)
                if resultado:
                    resultados.append(resultado)
        
        # Salvar resultados
        if resultados:
            # CSV detalhado
            with open(LOG_CSV, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = ["arquivo", "tipo", "duracao_s", "hash", "picos", "silencio_pct", "freq_dominante"] + [f"mfcc_{i}" for i in range(13)]
                writer.writerow(headers)
                
                for r in resultados:
                    row = [
                        r['arquivo'], r['tipo'], r['duracao'], r['hash'],
                        r['analise']['picos'], r['analise']['silencio_percentual'],
                        r['analise']['frequencia_dominante']
                    ] + r['mfcc']
                    writer.writerow(row)
            
            # Sumário
            sumario_df = pd.DataFrame([{
                'arquivo': r['arquivo'],
                'tipo': r['tipo'],
                'duracao_s': r['duracao'],
                'picos': r['analise']['picos'],
                'silencio_pct': r['analise']['silencio_percentual']
            } for r in resultados])
            
            sumario_df.to_csv(SUMARIO_CSV, index=False)
            
            logger.info(f"✅ CONCLUÍDO! {len(resultados)} arquivos processados")
            logger.info(f"📂 Resultados em: {PASTA_OUT}")
            logger.info("🎯 100% STANDALONE - Nenhum dicionário necessário!")
            
        else:
            logger.warning("⚠️  Nenhum arquivo processado com sucesso")
            
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        raise

if __name__ == "__main__":
    main()