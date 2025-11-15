#!/bin/bash
# ============================================================================
# Script para iniciar Jupyter Notebook no WSL e abrir no navegador Windows
# ============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="/home/ericobon/insightesfera/PORTFOLIO_ACADEMICO/pi4v10"
cd "$PROJECT_DIR"

# Ativar ambiente virtual
source venv/bin/activate

echo -e "${BLUE}🦟 Dengue ML - Iniciando Jupyter Notebook${NC}\n"

echo -e "${GREEN}✅ Ambiente virtual ativado${NC}"
echo -e "${GREEN}✅ Kernel disponível: Python (Dengue ML)${NC}\n"

echo -e "${YELLOW}📋 INSTRUÇÕES:${NC}"
echo -e "   1. O Jupyter vai iniciar no WSL"
echo -e "   2. Copie o link que aparecerá (começa com http://localhost:8888...)"
echo -e "   3. Cole no navegador do Windows (Chrome, Edge, etc)"
echo -e "   4. No Jupyter: Kernel → Change Kernel → Python (Dengue ML)"
echo -e ""
echo -e "${BLUE}Iniciando Jupyter...${NC}\n"

# Iniciar Jupyter sem tentar abrir navegador
jupyter notebook --no-browser modelo_dengue_final_optuna.ipynb
