# DrBIJ - Audio Forensic Analysis

## Overview
Advanced audio forensic analysis script with cross-platform compatibility and enhanced error handling.

## ✨ Recent Improvements (Cross-Platform Compatibility)

### 🔧 Environment-Specific Fixes
- **Enhanced dependency checking** with clear error messages
- **FFmpeg validation** across different operating systems
- **Platform-specific error handling** for Windows, Linux, and macOS
- **Optional dependency support** (Whisper can be disabled)
- **Improved encoding compatibility** for different locales

### 🌍 Cross-Platform Support
- ✅ Windows 10/11 (including WSL)
- ✅ Linux (Ubuntu, Debian, CentOS, RHEL)
- ✅ macOS (Intel and Apple Silicon)
- ✅ Python 3.8+ compatibility

### 🚀 Key Features
- **Audio format conversion** with FFmpeg integration
- **MFCC fingerprinting** for audio analysis
- **Speech transcription** with Whisper (optional)
- **Forensic reporting** with PDF generation
- **Memory management** and performance optimization
- **Progress tracking** with detailed logging

## 🛠️ Installation

### Prerequisites
1. **Python 3.8+** 
2. **FFmpeg** (system dependency)

### Quick Setup

#### Windows
```bash
# Install FFmpeg
winget install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install ffmpeg python3-pip

# Install Python dependencies
pip3 install -r requirements.txt
```

#### macOS
```bash
# Install FFmpeg
brew install ffmpeg

# Install Python dependencies
pip3 install -r requirements.txt
```

## 🚀 Usage

### Basic Usage
```bash
python3 audio_forensic_improved.py
```

### Directory Structure
```
DrBIJ/
├── audio_forensic_improved.py    # Main script
├── requirements.txt              # Python dependencies
├── INA/                         # Input audio files
├── OUTA/                        # Output reports and analysis
├── liwc_categorias.json         # LIWC categories (optional)
└── mensagens_ge_london_categorizadas.csv  # Text data (optional)
```

### Input Formats Supported
- WAV, MP3, M4A, OPUS, FLAC, AAC

### Output Files
- **Detailed CSV reports** with audio analysis
- **PDF forensic reports** with spectrograms
- **SRT transcription files** (if Whisper available)
- **Audio spectrograms** (PNG format)

## 🔍 Features

### Audio Analysis
- **Hash calculation** for file integrity
- **Peak detection** and silence analysis
- **MFCC feature extraction** for audio fingerprinting
- **Spectrogram generation** for visual analysis

### Transcription & Categorization
- **Whisper integration** for speech-to-text
- **LIWC categorization** for emotional analysis
- **SRT subtitle generation** with timestamps
- **Automatic audio categorization**

### Reporting
- **PDF forensic reports** with complete analysis
- **CSV data export** for further processing
- **Progress tracking** with detailed logging
- **Memory usage monitoring**

## 🐛 Troubleshooting

### Common Issues

#### FFmpeg Not Found
```
❌ FFmpeg não disponível: FFmpeg não encontrado no PATH do sistema
```
**Solution**: Install FFmpeg using platform-specific instructions above

#### Missing Dependencies
```
❌ Erro: numpy não encontrado. Instale com: pip install numpy
```
**Solution**: Install missing packages with pip

#### Whisper Unavailable
```
⚠️  Aviso: Whisper não encontrado. Transcrição será desabilitada.
```
**Solution**: Install Whisper with `pip install openai-whisper` (optional)

#### Permission Errors
```
❌ Erro de permissão no diretório de saída
```
**Solution**: Check file permissions and run with appropriate privileges

## 📖 Documentation

- [Cross-Platform Compatibility Fixes](CROSS_PLATFORM_FIXES.md)
- [Setup Guide](SETUP_GUIDE.md)
- [Audio Forensic Analysis](audio_forensic_analysis.md)

## 🧪 Testing

Run the included test suite:
```bash
python3 test_environment_fixes.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test across different platforms
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter environment-specific issues:
1. Check the [troubleshooting guide](CROSS_PLATFORM_FIXES.md)
2. Run the diagnostic test script
3. Create an issue with your platform details and error logs