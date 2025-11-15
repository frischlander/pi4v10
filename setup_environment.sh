#!/bin/bash
# ============================================================================
# Setup do Ambiente Virtual para Projeto Dengue ML
# ============================================================================

set -e

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="/home/ericobon/insightesfera/PORTFOLIO_ACADEMICO/pi4v10"
VENV_DIR="$PROJECT_DIR/venv"
KERNEL_NAME="dengue_ml"

echo -e "${BLUE}"
cat << "BANNER"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🦟  DENGUE ML - SETUP DE AMBIENTE VIRTUAL                  ║
║   Healthcare Machine Learning Project                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}\n"

# ============================================================================
# 1. VERIFICAR PYTHON
# ============================================================================

echo -e "${BLUE}🔍 Verificando Python...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 não encontrado!${NC}"
    echo -e "${YELLOW}Instale Python 3.8+ antes de continuar${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python encontrado: $PYTHON_VERSION${NC}"

# ============================================================================
# 2. CRIAR AMBIENTE VIRTUAL
# ============================================================================

echo -e "\n${BLUE}📦 Criando ambiente virtual...${NC}"

cd "$PROJECT_DIR"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Ambiente virtual já existe em $VENV_DIR${NC}"
    read -p "Deseja recriar? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Removendo ambiente antigo...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${BLUE}Usando ambiente existente...${NC}"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Ambiente virtual criado em: $VENV_DIR${NC}"
fi

# ============================================================================
# 3. ATIVAR E ATUALIZAR PIP
# ============================================================================

echo -e "\n${BLUE}🔄 Ativando ambiente virtual...${NC}"

source "$VENV_DIR/bin/activate"

echo -e "${GREEN}✅ Ambiente ativado!${NC}"

echo -e "\n${BLUE}⬆️  Atualizando pip...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✅ pip atualizado${NC}"

# ============================================================================
# 4. INSTALAR DEPENDÊNCIAS
# ============================================================================

echo -e "\n${BLUE}📥 Instalando dependências do requirements.txt...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt não encontrado!${NC}"
    exit 1
fi

echo -e "${YELLOW}⏳ Isso pode levar alguns minutos...${NC}\n"

pip install -r requirements.txt

echo -e "\n${GREEN}✅ Dependências instaladas!${NC}"

# ============================================================================
# 5. CONFIGURAR KERNEL DO JUPYTER
# ============================================================================

echo -e "\n${BLUE}🎯 Configurando kernel do Jupyter...${NC}"

# Instalar ipykernel se não estiver
pip install ipykernel > /dev/null 2>&1

# Registrar kernel
python3 -m ipykernel install --user --name="$KERNEL_NAME" --display-name="Python (Dengue ML)"

echo -e "${GREEN}✅ Kernel '$KERNEL_NAME' registrado no Jupyter!${NC}"

# ============================================================================
# 6. VERIFICAR INSTALAÇÃO
# ============================================================================

echo -e "\n${BLUE}🔍 Verificando instalações principais...${NC}\n"

# Lista de pacotes críticos
CRITICAL_PACKAGES=("pandas" "numpy" "scikit-learn" "xgboost" "catboost" "shap" "matplotlib" "seaborn" "plotly" "jupyter")

all_ok=true
for package in "${CRITICAL_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "N/A")
        echo -e "   ${GREEN}✅ $package${NC} ($version)"
    else
        echo -e "   ${RED}❌ $package${NC}"
        all_ok=false
    fi
done

if [ "$all_ok" = true ]; then
    echo -e "\n${GREEN}✅ Todas as dependências críticas instaladas!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Algumas dependências falharam. Execute novamente:${NC}"
    echo -e "${YELLOW}   pip install -r requirements.txt${NC}"
fi

# ============================================================================
# 7. CRIAR SCRIPT DE ATIVAÇÃO RÁPIDA
# ============================================================================

echo -e "\n${BLUE}📝 Criando script de ativação rápida...${NC}"

cat > "$PROJECT_DIR/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Ativação rápida do ambiente virtual

PROJECT_DIR="/home/ericobon/insightesfera/PORTFOLIO_ACADEMICO/pi4v10"
source "$PROJECT_DIR/venv/bin/activate"

echo "🦟 Ambiente Dengue ML ativado!"
echo "Para executar Jupyter: jupyter notebook"
ACTIVATE_EOF

chmod +x "$PROJECT_DIR/activate.sh"

echo -e "${GREEN}✅ Script de ativação criado: activate.sh${NC}"

# ============================================================================
# 8. CRIAR ARQUIVO .gitignore
# ============================================================================

echo -e "\n${BLUE}📝 Criando .gitignore...${NC}"

cat > "$PROJECT_DIR/.gitignore" << 'GITIGNORE_EOF'
# Virtual Environment
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Models and outputs
*.pkl
*.h5
*.joblib
best_model_*.pkl

# Data (comentar se quiser versionar)
# *.csv
# *.parquet

# Visualizations
*.png
*.jpg
*.jpeg
*.pdf

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
GITIGNORE_EOF

echo -e "${GREEN}✅ .gitignore criado${NC}"

# ============================================================================
# MENSAGEM FINAL
# ============================================================================

echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  ✅  AMBIENTE CONFIGURADO COM SUCESSO!                        ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}📋 Próximos passos:${NC}\n"

echo -e "1. ${YELLOW}Ativar o ambiente virtual:${NC}"
echo -e "   ${GREEN}source $VENV_DIR/bin/activate${NC}"
echo -e "   ${BLUE}ou${NC}"
echo -e "   ${GREEN}source activate.sh${NC}"
echo -e ""

echo -e "2. ${YELLOW}Executar Jupyter Notebook:${NC}"
echo -e "   ${GREEN}jupyter notebook dengue_prediction_advanced.ipynb${NC}"
echo -e ""

echo -e "3. ${YELLOW}No Jupyter, selecione o kernel:${NC}"
echo -e "   ${GREEN}Kernel → Change Kernel → Python (Dengue ML)${NC}"
echo -e ""

echo -e "4. ${YELLOW}Para desativar o ambiente:${NC}"
echo -e "   ${GREEN}deactivate${NC}"
echo -e ""

echo -e "${BLUE}📚 Arquivos importantes:${NC}"
echo -e "   - ${GREEN}dengue_prediction_advanced.ipynb${NC}  (Análise completa - RECOMENDADO)"
echo -e "   - ${GREEN}requirements.txt${NC}                  (Dependências)"
echo -e "   - ${GREEN}README_DENGUE_ML.md${NC}               (Documentação)"
echo -e "   - ${GREEN}activate.sh${NC}                       (Ativação rápida)"
echo -e ""

echo -e "${BLUE}🎯 Kernel Jupyter registrado:${NC}"
echo -e "   ${GREEN}Nome: Python (Dengue ML)${NC}"
echo -e "   ${GREEN}ID: $KERNEL_NAME${NC}"
echo -e ""

echo -e "${BLUE}🦟 Happy ML! Em saúde, Recall > tudo!${NC}\n"
