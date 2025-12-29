#!/bin/bash

# Setup script for Conversation Evaluation System
# This script prepares the environment and data

echo "🚀 Setting up Conversation Evaluation System..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p data/raw data/processed data/results logs models
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🐍 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Install requirements
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Check if Facets_Assignment.csv exists
if [ ! -f "data/raw/Facets_Assignment.csv" ]; then
    echo -e "${YELLOW}⚠️  Facets_Assignment.csv not found in data/raw/${NC}"
    echo -e "${YELLOW}   Please add the facets file to data/raw/Facets_Assignment.csv${NC}"
    echo ""
else
    echo -e "${GREEN}✅ Facets file found${NC}"
    echo ""
    
    # Process facets
    echo -e "${BLUE}🧹 Processing facets...${NC}"
    python src/preprocessing/facet_cleaner.py
    echo -e "${GREEN}✅ Facets processed${NC}"
    echo ""
fi

# Generate sample conversations
echo -e "${BLUE}💬 Generating sample conversations...${NC}"
python src/preprocessing/conversation_generator.py
echo -e "${GREEN}✅ Conversations generated${NC}"
echo ""

# Run evaluation (optional - can be slow)
read -p "Do you want to run the evaluation pipeline now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🔍 Running evaluation pipeline...${NC}"
    echo -e "${YELLOW}⚠️  This may take several minutes...${NC}"
    python src/pipeline/evaluation_pipeline.py
    echo -e "${GREEN}✅ Evaluation complete${NC}"
    echo ""
fi

# Final instructions
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To start the UI:${NC}"
echo -e "  streamlit run src/ui/app.py"
echo ""
echo -e "${BLUE}To run with Docker:${NC}"
echo -e "  docker-compose up --build"
echo ""
echo -e "${BLUE}To run evaluation manually:${NC}"
echo -e "  python src/pipeline/evaluation_pipeline.py"
echo ""
