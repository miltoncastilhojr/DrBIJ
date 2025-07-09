#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Forensic Analysis Script - Improved Version
================================================
Enhanced version with bug fixes and performance improvements
Cross-platform compatibility improvements
"""

import os
import sys
import platform
import shutil
import json
import csv
import hashlib
import subprocess
import gc
import functools
import logging
from typing import Dict, List, Tuple, Optional

# Try to import required packages with better error handling
try:
    import numpy as np
except ImportError as e:
    print(f"❌ Erro: numpy não encontrado. Instale com: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for better compatibility
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ Erro: matplotlib não encontrado. Instale com: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.io import wavfile
    from scipy.signal import spectrogram, find_peaks
    from scipy.fftpack import dct
except ImportError as e:
    print(f"❌ Erro: scipy não encontrado. Instale com: pip install scipy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print(f"❌ Erro: pandas não encontrado. Instale com: pip install pandas")
    sys.exit(1)

try:
    import psutil
except ImportError as e:
    print(f"❌ Erro: psutil não encontrado. Instale com: pip install psutil")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Erro: tqdm não encontrado. Instale com: pip install tqdm")
    sys.exit(1)

try:
    from fpdf import FPDF
except ImportError as e:
    print(f"❌ Erro: fpdf2 não encontrado. Instale com: pip install fpdf2")
    sys.exit(1)

# Optional imports with fallback
WHISPER_AVAILABLE = True
try:
    import whisper
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"⚠️  Aviso: Whisper não encontrado. Transcrição será desabilitada.")
    print(f"   Instale com: pip install openai-whisper")
    
    # Create a dummy whisper module for compatibility
    class DummyWhisper:
        @staticmethod
        def load_model(*args, **kwargs):
            return None
        @staticmethod
        def transcribe(*args, **kwargs):
            return {"text": "[Transcrição indisponível - Whisper não instalado]"}
    
    whisper = DummyWhisper()

# === CONFIGURAÇÕES ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE_DIR, "INA")
PASTA_OUT = os.path.join(BASE_DIR, "OUTA")
CATEGORIAS_JSON = "liwc_categorias.json"
TEXTO_CSV = "mensagens_ge_london_categorizadas.csv"
SUMARIO_CSV = os.path.join(PASTA_OUT, "sumario_relatorio_geral.csv")
SUMARIO_PDF = os.path.join(PASTA_OUT, "sumario_relatorio_geral.pdf")
LOG_CSV = os.path.join(PASTA_OUT, "relatorio_forense.csv")

# Create output directory
os.makedirs(PASTA_OUT, exist_ok=True)

# === CONFIGURAÇÃO APRIMORADA ===
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
        'cleanup_interval': 10
    }
}

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PASTA_OUT, 'forensic_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === FUNÇÕES DE VERIFICAÇÃO ===
def verificar_plataforma():
    """Check platform and provide platform-specific configurations"""
    sistema = platform.system().lower()
    plataforma_info = {
        'sistema': sistema,
        'versao': platform.release(),
        'arquitetura': platform.machine(),
        'python_version': platform.python_version(),
        'is_windows': sistema == 'windows',
        'is_linux': sistema == 'linux',
        'is_macos': sistema == 'darwin'
    }
    
    logger.info(f"🖥️  Sistema detectado: {plataforma_info['sistema']} {plataforma_info['versao']}")
    logger.info(f"🐍 Python {plataforma_info['python_version']} ({plataforma_info['arquitetura']})")
    
    return plataforma_info

def verificar_ffmpeg():
    """Check if FFmpeg is available and accessible"""
    try:
        # Try to find ffmpeg in PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            return False, "FFmpeg não encontrado no PATH do sistema"
        
        # Test if ffmpeg is actually working
        result = subprocess.run(
            [ffmpeg_path, '-version'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            logger.info(f"✅ FFmpeg encontrado: {version_line}")
            return True, ffmpeg_path
        else:
            return False, f"FFmpeg não está funcionando corretamente: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "FFmpeg não responde (timeout)"
    except Exception as e:
        return False, f"Erro ao verificar FFmpeg: {str(e)}"

def verificar_dependencias():
    """Check if all required dependencies are available"""
    plataforma = verificar_plataforma()
    
    # Check Python packages
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'pandas', 
        'psutil', 'fpdf', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        raise ImportError(f"Pacotes em falta: {', '.join(missing)}")
    
    # Check FFmpeg
    ffmpeg_ok, ffmpeg_info = verificar_ffmpeg()
    if not ffmpeg_ok:
        error_msg = f"❌ FFmpeg não disponível: {ffmpeg_info}\n"
        if plataforma['is_windows']:
            error_msg += "   Baixe FFmpeg de: https://ffmpeg.org/download.html\n"
            error_msg += "   Ou instale com: winget install ffmpeg"
        elif plataforma['is_linux']:
            error_msg += "   Instale com: sudo apt-get install ffmpeg (Ubuntu/Debian)\n"
            error_msg += "   Ou: sudo yum install ffmpeg (CentOS/RHEL)"
        elif plataforma['is_macos']:
            error_msg += "   Instale com: brew install ffmpeg"
        
        raise RuntimeError(error_msg)
    
    # Check Whisper availability
    if not WHISPER_AVAILABLE:
        logger.warning("⚠️  Whisper não disponível - transcrição será limitada")
        logger.warning("   Instale com: pip install openai-whisper")
    
    # Check write permissions
    try:
        test_file = os.path.join(PASTA_OUT, 'test_write.tmp')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('test')
        os.remove(test_file)
        logger.info("✅ Permissões de escrita verificadas")
    except Exception as e:
        raise PermissionError(f"Erro de permissão no diretório de saída: {e}")
    
    logger.info("✅ Todas as dependências verificadas")
    return plataforma

def verificar_memoria_detalhada():
    """Enhanced memory monitoring"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()
    
    if mem_percent > CONFIG['processing']['max_memory_percent']:
        gc.collect()  # Force garbage collection
        mem_percent = process.memory_percent()  # Check again after cleanup
        
        if mem_percent > 90:
            raise MemoryError(f"Uso crítico de memória: {mem_percent:.1f}%")
    
    return {
        'percent': mem_percent,
        'rss': mem_info.rss / 1024 / 1024,  # MB
        'vms': mem_info.vms / 1024 / 1024   # MB
    }

def validar_caminho_arquivo(path: str) -> str:
    """Validate file paths to prevent directory traversal"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    resolved = os.path.realpath(path)
    expected_base = os.path.realpath(PASTA)
    
    if not resolved.startswith(expected_base):
        raise ValueError("Caminho de arquivo inválido - fora do diretório permitido")
    
    return resolved

# === FUNÇÃO DE CONVERSÃO APRIMORADA ===
def converter_audio_safe(input_path: str, output_path: str, plataforma_info: Dict = None) -> bool:
    """Enhanced audio conversion with better error handling and platform support"""
    try:
        # Validate input file
        validar_caminho_arquivo(input_path)
        
        # Get FFmpeg path (platform-specific)
        ffmpeg_cmd = 'ffmpeg'
        if plataforma_info and plataforma_info.get('is_windows'):
            # On Windows, might need .exe extension
            if shutil.which('ffmpeg.exe'):
                ffmpeg_cmd = 'ffmpeg.exe'
        
        # Build command with cross-platform compatibility
        command = [
            ffmpeg_cmd, '-y',  # Overwrite output files
            '-i', input_path,
            '-ar', str(CONFIG['audio']['sample_rate']),
            '-ac', str(CONFIG['audio']['channels']),
            '-acodec', 'pcm_s16le',  # Specify codec
            '-loglevel', 'error',  # Reduce verbose output
            output_path
        ]
        
        # Platform-specific adjustments
        env = os.environ.copy()
        if plataforma_info and plataforma_info.get('is_windows'):
            # On Windows, ensure proper encoding
            env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=300,  # 5 minute timeout
            env=env,
            encoding='utf-8',  # Explicit encoding
            errors='replace'  # Handle encoding errors gracefully
        )
        
        # Verify output file was created and has content
        if not os.path.exists(output_path):
            logger.error(f"❌ Arquivo de saída não foi criado: {output_path}")
            return False
        
        if os.path.getsize(output_path) == 0:
            logger.error(f"❌ Arquivo de saída está vazio: {output_path}")
            return False
        
        logger.info(f"✅ Conversão bem-sucedida: {os.path.basename(input_path)}")
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        logger.error(f"❌ FFmpeg falhou para {input_path}: {error_msg}")
        
        # Provide platform-specific troubleshooting
        if plataforma_info:
            if plataforma_info.get('is_windows') and "codec" in error_msg.lower():
                logger.error("   Dica: Verifique se sua versão do FFmpeg suporta PCM")
            elif "permission" in error_msg.lower():
                logger.error("   Dica: Verifique permissões de arquivo e diretório")
        
        return False
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout na conversão de {input_path}")
        logger.error("   Arquivo pode ser muito grande ou corrompido")
        return False
        
    except FileNotFoundError:
        logger.error(f"❌ FFmpeg não encontrado. Instale FFmpeg primeiro.")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado na conversão de {input_path}: {e}")
        return False

# === CARREGAR ARQUIVOS PROCESSADOS ===
def carregar_arquivos_processados() -> set:
    """Load list of already processed files"""
    arquivos_processados = set()
    if os.path.exists(LOG_CSV):
        try:
            df_log = pd.read_csv(LOG_CSV)
            arquivos_processados = set(df_log['arquivo'].astype(str).str.strip().tolist())
            logger.info(f"📋 Carregados {len(arquivos_processados)} arquivos já processados")
        except Exception as e:
            logger.warning(f"⚠️  Erro ao ler relatorio_forense.csv: {e}")
    return arquivos_processados

# === INICIALIZAR WHISPER COM CACHE ===
@functools.lru_cache(maxsize=1)
def get_whisper_model():
    """Get Whisper model with caching and fallback handling"""
    if not WHISPER_AVAILABLE:
        logger.warning("🤖 Whisper não disponível - retornando modelo dummy")
        return None
    
    try:
        logger.info("🤖 Carregando modelo Whisper...")
        model = whisper.load_model(
            CONFIG['whisper']['model'], 
            device=CONFIG['whisper']['device']
        )
        logger.info("✅ Modelo Whisper carregado")
        return model
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Whisper: {e}")
        logger.warning("   Continuando sem transcrição...")
        return None

# === CARREGAR CONFIGURAÇÕES ===
def carregar_configuracoes():
    """Load LIWC categories and text data with improved encoding handling"""
    # Load LIWC categories
    try:
        # Try different encodings for cross-platform compatibility
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        liwc_categorias = {}
        
        for encoding in encodings_to_try:
            try:
                with open(CATEGORIAS_JSON, "r", encoding=encoding) as f:
                    liwc_categorias = json.load(f)
                logger.info(f"✅ Categorias LIWC carregadas (encoding: {encoding})")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except FileNotFoundError:
                break
        
        if not liwc_categorias and os.path.exists(CATEGORIAS_JSON):
            logger.warning("⚠️  Erro de encoding no arquivo LIWC, usando categorias vazias")
            
    except FileNotFoundError:
        logger.warning("⚠️  Arquivo LIWC não encontrado, usando categorias vazias")
        liwc_categorias = {}
    except Exception as e:
        logger.error(f"❌ Erro ao carregar LIWC: {e}")
        liwc_categorias = {}
    
    # Load text data
    try:
        if os.path.exists(TEXTO_CSV):
            # Try different encodings for CSV file
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            texto_df = None
            
            for encoding in encodings_to_try:
                try:
                    texto_df = pd.read_csv(TEXTO_CSV, encoding=encoding)
                    logger.info(f"✅ Base de texto carregada: {len(texto_df)} entradas (encoding: {encoding})")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if texto_df is None:
                logger.warning("⚠️  Erro de encoding no arquivo CSV, criando nova base")
                texto_df = pd.DataFrame(columns=["timestamp", "mensagem", "categoria"])
        else:
            texto_df = pd.DataFrame(columns=["timestamp", "mensagem", "categoria"])
            logger.info("ℹ️  Base de texto não encontrada, criando nova")
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar base de texto: {e}")
        texto_df = pd.DataFrame(columns=["timestamp", "mensagem", "categoria"])
    
    return liwc_categorias, texto_df

# === FUNÇÃO DE HASH APRIMORADA ===
def calcular_hash_seguro(path: str) -> str:
    """Calculate SHA256 hash with enhanced security"""
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(CONFIG['audio']['chunk_size']), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"❌ Erro ao calcular hash de {path}: {e}")
        raise

# === FUNÇÃO MFCC OTIMIZADA ===
def extract_mfcc_otimizado(signal: np.ndarray, rate: int, num_ceps: int = 13, nfft: int = 512) -> np.ndarray:
    """Optimized MFCC extraction with better error handling"""
    try:
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        # Frame settings
        frame_len = int(rate * 0.025)  # 25ms
        frame_step = int(rate * 0.01)  # 10ms
        signal_len = len(emphasized)
        
        # Calculate number of frames
        num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step)) + 1
        
        # Pad signal
        pad_signal = np.append(emphasized, np.zeros((num_frames * frame_step + frame_len - signal_len)))
        
        # Create frames
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_len)
        
        # FFT and power spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = (1.0 / nfft) * (mag_frames ** 2)
        
        # Mel filter bank
        nfilt = 26
        mel_points = np.linspace(0, 2595 * np.log10(1 + (rate / 2) / 700), nfilt + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_indices = np.floor((nfft + 1) * hz_points / rate).astype(int)
        
        fbank = np.zeros((nfilt, int(nfft / 2 + 1)))
        for m in range(1, nfilt + 1):
            for k in range(bin_indices[m - 1], bin_indices[m]):
                fbank[m - 1, k] = (k - bin_indices[m - 1]) / (bin_indices[m] - bin_indices[m - 1])
            for k in range(bin_indices[m], bin_indices[m + 1]):
                fbank[m - 1, k] = (bin_indices[m + 1] - k) / (bin_indices[m + 1] - bin_indices[m])
        
        # Apply filter bank
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        log_fb = 20 * np.log10(filter_banks)
        
        # DCT
        mfcc = dct(log_fb, type=2, axis=1, norm='ortho')[:, :num_ceps]
        
        return np.mean(mfcc, axis=0)
        
    except Exception as e:
        logger.error(f"❌ Erro na extração MFCC: {e}")
        return np.zeros(num_ceps)

# === CATEGORIZAÇÃO APRIMORADA ===
def categorizar_audio(texto: str, liwc_categorias: Dict) -> str:
    """Enhanced audio categorization"""
    if not texto.strip():
        return "vazio"
    
    texto_lower = texto.lower()
    for categoria, palavras in liwc_categorias.items():
        if any(palavra in texto_lower for palavra in palavras):
            return categoria
    
    return "outro"

# === FUNÇÃO DE TEMPO SRT CORRIGIDA ===
def formatar_tempo(segundos: float) -> str:
    """Fixed time formatting for SRT files"""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos_int = int(segundos % 60)
    milissegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02}:{minutos:02}:{segundos_int:02},{milissegundos:03}"

# === IDENTIFICAÇÃO DE PADRÕES ===
def identificar_padroes_textuais(texto_df: pd.DataFrame) -> Dict:
    """Identify emotional patterns in text data"""
    if texto_df.empty:
        return {}
    
    contagem = texto_df["categoria"].value_counts().to_dict()
    
    try:
        plt.figure(figsize=(12, 8))
        plt.bar(contagem.keys(), contagem.values(), color='skyblue')
        plt.title("Padrões Emocionais nas Mensagens", fontsize=14)
        plt.xlabel("Categoria")
        plt.ylabel("Frequência")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_OUT, "padroes_emocionais_textuais.png"), dpi=300)
        plt.close()
        logger.info("✅ Gráfico de padrões emocionais gerado")
    except Exception as e:
        logger.error(f"❌ Erro ao gerar gráfico de padrões: {e}")
    
    return contagem

# === GERAÇÃO DE PDF APRIMORADA ===
def gerar_pdf_aprimorado(arquivo: str, duracao: float, hash_val: str, picos: int, 
                        sils: int, fingerprint: List, transcricao: str, 
                        categoria: str, padroes: Dict):
    """Enhanced PDF generation with better error handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(0, 10, f"Relatório Forense - {arquivo}", ln=True, align='C')
        
        pdf.set_font("Arial", size=10)
        pdf.ln(5)
        
        # Informações básicas
        info_text = f"""
Duração: {duracao:.2f} segundos
Hash SHA256: {hash_val}
Categoria: {categoria}
Picos detectados: {picos}
Zonas de silêncio: {sils}
Fingerprint MFCC: {fingerprint}

Transcrição:
{transcricao[:500]}{"..." if len(transcricao) > 500 else ""}

Padrões emocionais identificados:
{padroes}
        """
        
        pdf.multi_cell(0, 5, info_text.strip())
        
        # Adicionar espectrograma se existir
        espectro_path = os.path.join(PASTA_OUT, f"{arquivo}_espectrograma.png")
        if os.path.exists(espectro_path):
            try:
                pdf.add_page()
                pdf.set_font("Arial", 'B', size=12)
                pdf.cell(0, 10, "Espectrograma", ln=True, align='C')
                pdf.image(espectro_path, x=10, w=190)
            except Exception as e:
                logger.warning(f"⚠️  Erro ao adicionar espectrograma ao PDF: {e}")
        
        # Salvar PDF
        pdf_path = os.path.join(PASTA_OUT, f"{arquivo}_relatorio.pdf")
        pdf.output(pdf_path)
        logger.info(f"✅ PDF gerado: {pdf_path}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar PDF para {arquivo}: {e}")

# === PROCESSAMENTO PRINCIPAL APRIMORADO ===
def processar_audio_aprimorado(path: str, pbar: tqdm, liwc_categorias: Dict, 
                              total_files: int, plataforma_info: Dict = None) -> Optional[Dict]:
    """Enhanced audio processing with better error handling and progress tracking"""
    try:
        # Verificar memória antes de processar
        mem_info = verificar_memoria_detalhada()
        
        base = os.path.splitext(os.path.basename(path))[0]
        converted_path = os.path.join(PASTA_OUT, base + ".wav")
        
        # Atualizar progresso
        pbar.set_description(f"🔄 Convertendo: {base}")
        
        # Converter áudio com informações da plataforma
        if not converter_audio_safe(path, converted_path, plataforma_info):
            logger.error(f"❌ Falha na conversão: {base}")
            return None
        
        # Calcular hash e ler áudio
        pbar.set_description(f"🔍 Analisando: {base}")
        hash_val = calcular_hash_seguro(converted_path)
        
        try:
            sr, samples = wavfile.read(converted_path)
            if samples.ndim > 1:
                samples = samples[:, 0]
            duration = len(samples) / sr
        except Exception as e:
            logger.error(f"❌ Erro ao ler arquivo de áudio {base}: {e}")
            return None
        
        # Gerar espectrograma
        try:
            f, t, Sxx = spectrogram(samples, fs=sr, nperseg=512)
            fig, ax = plt.subplots(figsize=(12, 8))
            mappable = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                                   shading='gouraud', cmap="viridis")
            plt.colorbar(mappable, ax=ax, label="Intensidade [dB]")
            ax.set_title(f"Espectrograma - {base}")
            ax.set_xlabel("Tempo [s]")
            ax.set_ylabel("Frequência [Hz]")
            plt.tight_layout()
            plt.savefig(os.path.join(PASTA_OUT, base + "_espectrograma.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"⚠️  Erro ao gerar espectrograma para {base}: {e}")
        
        # Análise de áudio
        amp = np.abs(samples)
        peaks, _ = find_peaks(amp, height=np.percentile(amp, 90))
        silences = np.where(amp < np.percentile(amp, 2))[0]
        
        # Extrair MFCC
        fingerprint = extract_mfcc_otimizado(samples, sr)
        
        # Transcrição
        pbar.set_description(f"🗣️  Transcrevendo: {base}")
        try:
            model = get_whisper_model()
            if model is None:
                # Whisper não disponível
                transcricao = "[Transcrição indisponível - Whisper não instalado]"
                categoria = "whisper_indisponivel"
                result = {"segments": []}
            else:
                result = model.transcribe(
                    converted_path, 
                    language=CONFIG['whisper']['language'],
                    word_timestamps=True,
                    fp16=CONFIG['whisper']['fp16'],
                    beam_size=CONFIG['whisper']['beam_size']
                )
                transcricao = result["text"]
                categoria = categorizar_audio(transcricao, liwc_categorias)
                
                # Gerar arquivo SRT apenas se Whisper estiver disponível
                srt_path = os.path.join(PASTA_OUT, base + ".srt")
                with open(srt_path, "w", encoding="utf-8") as srt_file:
                    for i, segment in enumerate(result["segments"]):
                        start = segment["start"]
                        end = segment["end"]
                        text = segment["text"].strip()
                        srt_file.write(f"{i+1}\n")
                        srt_file.write(f"{formatar_tempo(start)} --> {formatar_tempo(end)}\n")
                        srt_file.write(f"{text}\n\n")
            
        except Exception as e:
            logger.error(f"❌ Erro na transcrição de {base}: {e}")
            transcricao = f"[Erro na transcrição: {str(e)}]"
            categoria = "erro_transcricao"
            result = {"segments": []}
        
        # Gerar PDF
        pbar.set_description(f"📄 Gerando relatório: {base}")
        padroes = {}  # Seria passado de fora em implementação completa
        gerar_pdf_aprimorado(base, duration, hash_val, len(peaks), 
                            len(silences), np.round(fingerprint, 2).tolist(), 
                            transcricao, categoria, padroes)
        
        # Limpeza de memória
        del result
        gc.collect()
        
        pbar.update(1)
        
        return {
            "arquivo": base,
            "duração_s": round(duration, 2),
            "picos": len(peaks),
            "zonas_silencio": len(silences),
            "hash": hash_val,
            "categoria": categoria,
            "transcricao": transcricao,
            "fingerprint": np.round(fingerprint, 2).tolist()
        }
        
    except Exception as e:
        logger.error(f"❌ Erro geral no processamento de {path}: {e}")
        pbar.update(1)
        return None

# === FUNÇÃO PRINCIPAL ===
def main():
    """Main execution function"""
    logger.info("🚀 Iniciando análise forense de áudio")
    
    try:
        # Verificar dependências e plataforma
        plataforma_info = verificar_dependencias()
        
        # Carregar configurações
        liwc_categorias, texto_df = carregar_configuracoes()
        arquivos_processados = carregar_arquivos_processados()
        
        # Encontrar arquivos para processar
        if not os.path.exists(PASTA):
            logger.error(f"❌ Diretório de entrada não encontrado: {PASTA}")
            logger.error(f"   Crie o diretório: {PASTA}")
            return
        
        files_to_process = [
            file for file in os.listdir(PASTA) 
            if file.lower().endswith(('.wav', '.m4a', '.mp3', '.opus', '.flac', '.aac'))
        ]
        
        # Filtrar arquivos já processados
        files_to_process = [
            file for file in files_to_process
            if os.path.splitext(file)[0] not in arquivos_processados
        ]
        
        if not files_to_process:
            logger.info("ℹ️  Nenhum arquivo novo para processar")
            return
        
        logger.info(f"📁 Encontrados {len(files_to_process)} arquivos para processar")
        
        # Processar arquivos
        resultados = []
        with tqdm(total=len(files_to_process), desc="Progresso Geral") as pbar:
            for file in files_to_process:
                file_path = os.path.join(PASTA, file)
                
                try:
                    resultado = processar_audio_aprimorado(
                        file_path, pbar, liwc_categorias, len(files_to_process), plataforma_info
                    )
                    if resultado:
                        resultados.append(resultado)
                except Exception as e:
                    logger.error(f"❌ Erro no processamento de {file}: {e}")
                    pbar.update(1)
        
        # Salvar resultados
        if resultados:
            # Salvar CSV detalhado com tratamento de encoding
            try:
                with open(LOG_CSV, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "arquivo", "duração_s", "picos", "zonas_silencio", 
                        "hash", "categoria"
                    ] + [f"mfcc_{i}" for i in range(13)])
                    
                    for r in resultados:
                        writer.writerow([
                            r["arquivo"], r["duração_s"], r["picos"], 
                            r["zonas_silencio"], r["hash"], r["categoria"]
                        ] + r["fingerprint"])
                
                logger.info(f"✅ CSV detalhado salvo: {LOG_CSV}")
            except Exception as e:
                logger.error(f"❌ Erro ao salvar CSV detalhado: {e}")
            
            # Salvar sumário com tratamento de encoding
            try:
                sumario_df = pd.DataFrame(resultados)
                sumario_df.to_csv(SUMARIO_CSV, index=False, encoding="utf-8")
                logger.info(f"✅ Sumário salvo: {SUMARIO_CSV}")
            except Exception as e:
                logger.error(f"❌ Erro ao salvar sumário: {e}")
            
            # Gerar padrões emocionais
            try:
                padroes = identificar_padroes_textuais(texto_df)
            except Exception as e:
                logger.error(f"❌ Erro ao gerar padrões emocionais: {e}")
            
            logger.info(f"✅ Processamento concluído! {len(resultados)} arquivos processados")
            logger.info(f"📂 Resultados salvos em: {PASTA_OUT}")
        else:
            logger.warning("⚠️  Nenhum arquivo foi processado com sucesso")
            logger.warning("   Verifique se:")
            logger.warning("   - Os arquivos de áudio estão no diretório correto")
            logger.warning("   - FFmpeg está instalado e funcionando")
            logger.warning("   - Há permissões de escrita no diretório de saída")
            
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        
        # Provide platform-specific troubleshooting
        if plataforma_info:
            logger.error("💡 Dicas de solução de problemas:")
            if plataforma_info.get('is_windows'):
                logger.error("   - Verifique se o FFmpeg está no PATH do Windows")
                logger.error("   - Execute como administrador se necessário")
            elif plataforma_info.get('is_linux'):
                logger.error("   - Verifique permissões: chmod +x audio_forensic_improved.py")
                logger.error("   - Instale dependências: sudo apt-get install ffmpeg")
            elif plataforma_info.get('is_macos'):
                logger.error("   - Instale FFmpeg: brew install ffmpeg")
                logger.error("   - Verifique permissões de segurança do macOS")
        
        raise

if __name__ == "__main__":
    main()