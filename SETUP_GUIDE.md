# Audio Forensic Analysis - Setup Guide

## Quick Start

### 1. System Requirements
- **Python**: 3.8 or higher
- **FFmpeg**: Must be installed and accessible via command line
- **Memory**: 8GB+ RAM recommended for large audio files
- **Storage**: Sufficient space for temporary and output files

### 2. Installation

#### Install FFmpeg (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Install Python Dependencies:
```bash
pip install -r requirements.txt
```

#### Verify Installation:
```bash
python audio_forensic_improved.py --help
```

### 3. Directory Structure
Create the following directories in your project folder:
```
project/
├── audio_forensic_improved.py
├── requirements.txt
├── INA/                    # Input audio files
├── OUTA/                   # Output files (auto-created)
├── liwc_categorias.json    # LIWC categories (optional)
└── mensagens_ge_london_categorizadas.csv  # Text data (optional)
```

### 4. Configuration Files

#### LIWC Categories (liwc_categorias.json):
```json
{
  "positive": ["alegria", "feliz", "bom", "ótimo", "excelente"],
  "negative": ["triste", "ruim", "péssimo", "horrível", "terrível"],
  "anger": ["raiva", "ódio", "irritado", "furioso", "zangado"],
  "fear": ["medo", "assustado", "nervoso", "ansioso", "preocupado"]
}
```

## Key Improvements Over Original Script

### 🐛 Critical Bug Fixes
1. **Fixed SRT time formatting** - Corrected milliseconds calculation
2. **Enhanced memory management** - Prevents memory overflow
3. **Improved error handling** - Better recovery from failures

### ⚡ Performance Enhancements
1. **Whisper model caching** - Reduces load time
2. **Optimized MFCC extraction** - Better performance
3. **Enhanced progress tracking** - More detailed status updates
4. **Memory monitoring** - Automatic cleanup when needed

### 🔒 Security Improvements
1. **File path validation** - Prevents directory traversal attacks
2. **Enhanced hash verification** - Better integrity checking
3. **Input sanitization** - Safer file processing

### 📊 Better Logging & Monitoring
1. **Comprehensive logging** - File and console output
2. **Memory usage tracking** - Real-time monitoring
3. **Detailed error reporting** - Better debugging information

## Usage Examples

### Basic Usage:
```bash
# Place audio files in INA/ directory
python audio_forensic_improved.py
```

### Processing Specific Formats:
The script supports: `.wav`, `.m4a`, `.mp3`, `.opus`, `.flac`, `.aac`

### Output Files:
- **Individual Reports**: `OUTA/{filename}_relatorio.pdf`
- **Spectrograms**: `OUTA/{filename}_espectrograma.png`
- **Subtitles**: `OUTA/{filename}.srt`
- **CSV Log**: `OUTA/relatorio_forense.csv`
- **Summary**: `OUTA/sumario_relatorio_geral.csv`

## Configuration Options

### Memory Management:
```python
CONFIG = {
    'processing': {
        'max_memory_percent': 85,  # Trigger cleanup at 85%
        'cleanup_interval': 10     # Force cleanup every 10 files
    }
}
```

### Whisper Settings:
```python
CONFIG = {
    'whisper': {
        'model': 'tiny',        # 'tiny', 'base', 'small', 'medium', 'large'
        'device': 'cpu',        # 'cpu' or 'cuda'
        'language': 'Portuguese',
        'fp16': True,           # Use half precision for speed
        'beam_size': 2          # Reduce for speed, increase for accuracy
    }
}
```

### Audio Processing:
```python
CONFIG = {
    'audio': {
        'sample_rate': 16000,   # Whisper optimal rate
        'channels': 1,          # Mono audio
        'chunk_size': 4096      # File I/O chunk size
    }
}
```

## Troubleshooting

### Common Issues:

#### FFmpeg Not Found:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### Memory Issues:
- Reduce `max_memory_percent` in config
- Use smaller Whisper model ('tiny' instead of 'base')
- Process files in smaller batches

#### Transcription Errors:
- Check audio file integrity
- Verify audio format is supported
- Try different Whisper model size

#### Permission Errors:
- Ensure write permissions to output directory
- Check file ownership and permissions

### Performance Optimization:

#### For Large Files:
1. Use `tiny` Whisper model for speed
2. Enable `fp16` for faster processing
3. Reduce `beam_size` for speed over accuracy

#### For Better Accuracy:
1. Use `base` or `small` Whisper model
2. Increase `beam_size` to 5
3. Ensure high-quality audio input

## Monitoring & Debugging

### Log Files:
- **Main Log**: `OUTA/forensic_analysis.log`
- **Console Output**: Real-time status updates
- **Progress Bar**: File-by-file processing status

### Memory Monitoring:
The script automatically monitors memory usage and will:
1. Force garbage collection at 85% usage
2. Raise error at 90% usage
3. Log memory statistics

### Error Recovery:
- Failed files are skipped and logged
- Processing continues with remaining files
- Detailed error information in logs

## Production Deployment

### Batch Processing:
```bash
# Process large batches overnight
nohup python audio_forensic_improved.py > processing.log 2>&1 &
```

### Scheduling:
```bash
# Add to crontab for regular processing
0 2 * * * cd /path/to/project && python audio_forensic_improved.py
```

### Resource Limits:
```bash
# Limit memory usage (4GB example)
ulimit -m 4194304
python audio_forensic_improved.py
```

## Contributing

### Code Standards:
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include comprehensive docstrings
- Add unit tests for new features

### Testing:
```bash
# Run with test data
python -m pytest tests/
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Ensure all dependencies are properly installed
4. Verify system requirements are met