# Git - Guia Simples para Subir os Arquivos

## 🎯 Objetivo: Subir todos os arquivos do projeto para o GitHub

### 📁 **Arquivos que temos:**
- `audio_forensic_standalone.py` ⭐ (Principal - RECOMENDADO)
- `audio_forensic_no_emotions.py` 
- `audio_forensic_improved.py`
- `requirements.txt`
- `SETUP_GUIDE.md`
- `AJUSTES_TECNICOS.md`
- `COMO_DESATIVAR_EMOCOES.md`
- `audio_forensic_analysis.md`

---

## 🚀 **Passo a Passo - Git Básico**

### 1️⃣ **Verificar Status**
```bash
git status
```
**O que mostra:** Arquivos novos/modificados

### 2️⃣ **Adicionar TODOS os Arquivos**
```bash
# Adiciona TUDO de uma vez
git add .

# OU adicionar arquivo por arquivo:
git add audio_forensic_standalone.py
git add requirements.txt
git add *.md
```

### 3️⃣ **Verificar o que foi Adicionado**
```bash
git status
```
**Deve mostrar:** Arquivos em verde (prontos para commit)

### 4️⃣ **Fazer Commit (Salvar Localmente)**
```bash
git commit -m "🎯 Audio Forensic Analysis - Versão Standalone Completa

✅ Script principal: audio_forensic_standalone.py (SEM dicionários)
✅ Suporte a áudio e vídeo
✅ Espectrogramas HD + relatórios com timestamps
✅ Detecção inteligente de picos/silêncios
✅ Documentação completa
✅ Bug SRT corrigido"
```

### 5️⃣ **Enviar para GitHub**
```bash
git push origin main
```

**OU se a branch for master:**
```bash
git push origin master
```

---

## 🔧 **Comandos Úteis**

### **Ver Histórico:**
```bash
git log --oneline
```

### **Ver Diferenças:**
```bash
git diff
```

### **Desfazer Última Modificação:**
```bash
git checkout -- nome_arquivo.py
```

### **Ver Branches:**
```bash
git branch
```

---

## 🆘 **Problemas Comuns**

### ❌ **"Nothing to commit"**
**Solução:**
```bash
git add .
git status  # Verificar se arquivos foram adicionados
```

### ❌ **"Repository not found"**
**Solução:**
```bash
# Verificar remote
git remote -v

# Se não tiver, adicionar:
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
```

### ❌ **"Permission denied"**
**Soluções:**
```bash
# Opção 1: HTTPS com token
git remote set-url origin https://SEU_TOKEN@github.com/SEU_USUARIO/SEU_REPO.git

# Opção 2: SSH (se configurado)
git remote set-url origin git@github.com:SEU_USUARIO/SEU_REPO.git
```

### ❌ **"Branch diverged"**
**Solução:**
```bash
# Puxar mudanças primeiro
git pull origin main

# Depois fazer push
git push origin main
```

---

## 🎯 **Sequência Completa - COPIE E COLE**

```bash
# 1. Ver status
git status

# 2. Adicionar tudo
git add .

# 3. Verificar
git status

# 4. Commit
git commit -m "🎯 Audio Forensic Analysis - Versão Standalone Completa"

# 5. Push
git push origin main
```

---

## 📋 **Estrutura Final no GitHub**

Depois do push, seu repositório terá:

```
seu-repositorio/
├── audio_forensic_standalone.py    ⭐ PRINCIPAL
├── audio_forensic_no_emotions.py   
├── audio_forensic_improved.py      
├── requirements.txt                
├── SETUP_GUIDE.md                  📖 Como usar
├── AJUSTES_TECNICOS.md            🔧 Melhorias técnicas
├── COMO_DESATIVAR_EMOCOES.md      ❌ Guia sem emoções
├── audio_forensic_analysis.md      📊 Análise completa
├── GIT_GUIA_SIMPLES.md            📚 Este guia
└── README.md                       📝 (se já existir)
```

---

## 💡 **Dicas Importantes**

### ✅ **SEMPRE Fazer Antes do Push:**
```bash
git status    # Ver o que mudou
git add .     # Adicionar tudo
git status    # Confirmar que está tudo verde
```

### ✅ **Mensagens de Commit Úteis:**
```bash
# ✅ Bom
git commit -m "🐛 Corrigido bug no timestamp SRT"
git commit -m "✨ Adicionado suporte a vídeo"
git commit -m "📝 Atualizada documentação"

# ❌ Ruim
git commit -m "update"
git commit -m "fix"
```

### ✅ **Verificar Se Subiu:**
1. Faça o push
2. Vá no GitHub no seu navegador
3. Atualize a página
4. Deve ver todos os arquivos lá

---

## 🚨 **Se Estiver MUITO Perdido**

### **Opção 1: Reset Completo**
```bash
# ⚠️ CUIDADO: Apaga mudanças locais
git reset --hard HEAD
git pull origin main
```

### **Opção 2: Ver Exatamente o Que Fazer**
```bash
git status
```
**Me mande a saída deste comando e eu te digo exatamente o que fazer!**

### **Opção 3: Interface Visual**
- **GitHub Desktop** (mais fácil)
- **VS Code** (extensão Git integrada)
- **SourceTree** (gratuito)

---

## 🎯 **AÇÃO IMEDIATA**

**Cole estes comandos no terminal:**

```bash
git status
git add .
git commit -m "🎯 Audio Forensic Analysis - Versão Standalone Completa

✅ Script standalone sem dependências externas
✅ Suporte completo a áudio e vídeo  
✅ Análise forense técnica profissional
✅ Documentação completa incluída"
git push origin main
```

**Se der erro, me mande a mensagem e eu ajudo a resolver!** 🛠️