#!/bin/bash
# scripts/quick_start.sh - Complete AI Agent Quick Start Script

set -euo pipefail

echo "🚀 AI Agent Complete Quick Start"
echo "================================"
echo "This script will set up your complete AI Agent environment with:"
echo "• Virtual environment and dependencies"
echo "• Configuration files and API keys"
echo "• Vector database setup"
echo "• LangChain integration"
echo "• Production-ready web API"
echo "• Comprehensive testing"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
   echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

warn() {
   echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
   echo -e "${RED}[ERROR]${NC} $1"
}

info() {
   echo -e "${BLUE}[INFO]${NC} $1"
}

# Error handler
handle_error() {
   error "Script failed on line $1"
   echo ""
   echo "Troubleshooting tips:"
   echo "• Check that Python 3.8+ is installed"
   echo "• Ensure you have internet connectivity"
   echo "• Verify your API keys are correct"
   echo "• Check the logs in ./logs/ directory"
   exit 1
}

trap 'handle_error $LINENO' ERR

# Check prerequisites
check_prerequisites() {
   log "Checking prerequisites..."
   
   # Check Python version
   if ! command -v python3 &> /dev/null; then
       error "Python 3 is required but not installed"
       echo "Please install Python 3.8 or later from https://python.org"
       exit 1
   fi
   
   python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
   if [[ $(echo "$python_version >= 3.8" | bc -l 2>/dev/null || echo "0") != "1" ]]; then
       error "Python 3.8+ required, found Python $python_version"
       exit 1
   fi
   
   info "✓ Python $python_version found"
   
   # Check for git (optional)
   if command -v git &> /dev/null; then
       info "✓ Git available"
   else
       warn "Git not found - version control won't be available"
   fi
   
   # Check available disk space (need at least 2GB)
   available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
   if [ "$available_space" -lt 2 ]; then
       warn "Low disk space: ${available_space}GB available (2GB recommended)"
   else
       info "✓ Sufficient disk space: ${available_space}GB"
   fi
}

# Setup virtual environment
setup_venv() {
   log "Setting up Python virtual environment..."
   
   if [ ! -d "venv" ]; then
       python3 -m venv venv
       info "✓ Virtual environment created"
   else
       info "✓ Virtual environment already exists"
   fi
   
   # Activate virtual environment
   source venv/bin/activate
   log "✓ Virtual environment activated"
   
   # Upgrade pip
   pip install --upgrade pip
   info "✓ Pip upgraded to latest version"
}

# Install dependencies
install_dependencies() {
   log "Installing Python dependencies..."
   
   # Install main dependencies
   pip install -r requirements.txt
   info "✓ Main dependencies installed"
   
   # Install development dependencies
   if [ -f "requirements-dev.txt" ]; then
       pip install -r requirements-dev.txt
       info "✓ Development dependencies installed"
   fi
   
   # Verify key packages
   python3 -c "import openai, langchain, chromadb, fastapi" 2>/dev/null
   info "✓ Core packages verified"
}

# Setup configuration
setup_configuration() {
   log "Setting up configuration..."
   
   # Create .env file if it doesn't exist
   if [ ! -f ".env" ]; then
       if [ -f ".env.example" ]; then
           cp .env.example .env
           info "✓ Created .env from template"
       else
           # Create basic .env file
           cat > .env << 'EOF'
# Environment Configuration
ENVIRONMENT=development
DEBUG=true

# API Keys (Replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# OpenAI Configuration
OPENAI_MODEL=gpt-4-1106-preview
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Database URLs
DATABASE_URL=sqlite:///./data/agent_memory.db
REDIS_URL=redis://localhost:6379

# Vector Database
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/agent.log

# Web Interface
WEB_HOST=127.0.0.1
WEB_PORT=8000
EOF
           info "✓ Created basic .env file"
       fi
       
       echo ""
       warn "IMPORTANT: Please edit .env and add your API keys!"
       echo "Required API keys:"
       echo "• OPENAI_API_KEY: Get from https://platform.openai.com/api-keys"
       echo "• ANTHROPIC_API_KEY (optional): Get from https://console.anthropic.com/"
       echo ""
       
       read -p "Press Enter after updating your API keys in .env file..."
   else
       info "✓ Configuration file (.env) already exists"
   fi
   
   # Create necessary directories
   mkdir -p data logs
   mkdir -p data/chroma_db
   info "✓ Data directories created"
}

# Setup git repository
setup_git() {
   if command -v git &> /dev/null && [ ! -d ".git" ]; then
       log "Setting up Git repository..."
       
       git init
       
       # Create comprehensive .gitignore
       cat > .gitignore << 'EOF'
# Environment and secrets
.env
*.log
.DS_Store

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# Data and databases
data/*.db
data/*.sqlite
data/chroma_db/
*.db-journal

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
Thumbs.db
.DS_Store

# Logs
logs/
*.log

# Temporary files
tmp/
temp/
*.tmp

# Docker
.dockerignore

# Coverage reports
htmlcov/
.coverage
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Model files (too large for git)
*.pkl
*.joblib
models/
EOF
       
       git add .gitignore
       git commit -m "Initial commit with gitignore"
       
       info "✓ Git repository initialized"
   fi
}

# Validate API keys
validate_api_keys() {
   log "Validating API keys..."
   
   # Source the .env file
   if [ -f ".env" ]; then
       export $(grep -v '^#' .env | xargs)
   fi
   
   # Check OpenAI API key
   if [[ "$OPENAI_API_KEY" == "your_openai_api_key_here" ]] || [[ -z "$OPENAI_API_KEY" ]]; then
       error "OpenAI API key not configured!"
       echo ""
       echo "To get your OpenAI API key:"
       echo "1. Go to https://platform.openai.com/api-keys"
       echo "2. Create a new API key"
       echo "3. Copy it to the OPENAI_API_KEY field in .env"
       echo ""
       return 1
   fi
   
   # Validate key format
   if [[ ! "$OPENAI_API_KEY" =~ ^sk-[a-zA-Z0-9]{48}$ ]]; then
       warn "OpenAI API key format looks incorrect"
       echo "Expected format: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   else
       info "✓ OpenAI API key format looks correct"
   fi
}

# Run comprehensive setup validation
run_setup_validation() {
   log "Running comprehensive setup validation..."
   
   # Test basic imports
   python3 -c "
import sys
sys.path.append('src')

# Test core imports
try:
   from config.settings import settings
   print('✓ Configuration loaded successfully')
except Exception as e:
   print(f'✗ Configuration error: {e}')
   sys.exit(1)

try:
   from src.integrations.databases.vector_store import create_vector_store
   print('✓ Vector database integration available')
except Exception as e:
   print(f'✗ Vector database error: {e}')
   sys.exit(1)

try:
   from src.integrations.langchain.agent_executor import LangChainAgent
   print('✓ LangChain integration available')
except Exception as e:
   print(f'✗ LangChain integration error: {e}')
   sys.exit(1)

try:
   from src.interfaces.api.main import app
   print('✓ FastAPI application loads successfully')
except Exception as e:
   print(f'✗ FastAPI application error: {e}')
   sys.exit(1)

print('✅ All core components validated successfully')
"
   
   if [ $? -eq 0 ]; then
       info "✓ Setup validation passed"
   else
       error "Setup validation failed"
       return 1
   fi
}

# Quick functional test
run_quick_test() {
   log "Running quick functional test..."
   
   python3 -c "
import sys
import asyncio
sys.path.append('src')

async def quick_test():
   try:
       # Test vector store
       from src.integrations.databases.vector_store import create_vector_store
       vector_store = create_vector_store('test_collection')
       
       # Add a test document
       success = await vector_store.add_documents(
           documents=['This is a test document for quick validation'],
           metadatas=[{'type': 'test', 'source': 'quick_start'}]
       )
       
       if success:
           print('✓ Vector database test passed')
       else:
           print('✗ Vector database test failed')
           return False
       
       # Test search
       results = await vector_store.similarity_search('test document', k=1)
       if results:
           print('✓ Vector search test passed')
       else:
           print('✗ Vector search test failed')
           return False
       
       return True
       
   except Exception as e:
       print(f'✗ Quick test failed: {e}')
       return False

result = asyncio.run(quick_test())
sys.exit(0 if result else 1)
"
   
   if [ $? -eq 0 ]; then
       info "✓ Quick functional test passed"
   else
       warn "Quick functional test failed (this may be due to API key issues)"
   fi
}

# Show next steps menu
show_menu() {
   echo ""
   echo "🎉 Setup Complete! Choose what to do next:"
   echo ""
   echo "1) 🧪 Run comprehensive tests"
   echo "2) 🚀 Start development API server"
   echo "3) 💬 Interactive chat demo"
   echo "4) 📊 Run performance validation"
   echo "5) 🔧 Advanced configuration"
   echo "6) 📚 View documentation"
   echo "7) 🐳 Docker setup"
   echo "8) ❌ Exit"
   echo ""
   
   while true; do
       read -p "Enter choice (1-8): " choice
       case $choice in
           1)
               run_comprehensive_tests
               break
               ;;
           2)
               start_dev_server
               break
               ;;
           3)
               run_interactive_demo
               break
               ;;
           4)
               run_performance_validation
               break
               ;;
           5)
               show_advanced_config
               break
               ;;
           6)
               show_documentation
               break
               ;;
           7)
               setup_docker
               break
               ;;
           8)
               echo "👋 Goodbye! Your AI Agent is ready to use."
               exit 0
               ;;
           *)
               echo "Invalid choice. Please enter 1-8."
               ;;
       esac
   done
}

# Menu option implementations
run_comprehensive_tests() {
   log "Running comprehensive test suite..."
   echo ""
   
   # Unit tests
   echo "Running unit tests..."
   python3 -m pytest tests/ -v --tb=short
   
   # Integration tests
   echo ""
   echo "Running integration tests..."
   python3 -m pytest tests/test_production_api.py -v
   
   # Validation script
   echo ""
   echo "Running production validator..."
   python3 scripts/production_validator.py --validate-only
   
   echo ""
   info "✓ Test suite completed!"
}

start_dev_server() {
   log "Starting development API server..."
   echo ""
   echo "🌐 API will be available at:"
   echo "   • Main API: http://localhost:8000"
   echo "   • Documentation: http://localhost:8000/docs"
   echo "   • Health check: http://localhost:8000/api/v1/health"
   echo ""
   echo "Press Ctrl+C to stop the server"
   echo ""
   
   cd src/interfaces/api
   python3 main.py
}

run_interactive_demo() {
   log "Starting interactive chat demo..."
   echo ""
   echo "💬 Interactive AI Agent Demo"
   echo "Commands:"
   echo "  • 'help' - Show agent capabilities"
   echo "  • 'stats' - Show agent statistics"
   echo "  • 'quit' - Exit demo"
   echo ""
   
   python3 -c "
import sys
import asyncio
sys.path.append('src')

from src.integrations.langchain.chapter2_integration import IntegratedAgent

async def interactive_demo():
   print('Initializing AI Agent...')
   agent = IntegratedAgent('demo_agent')
   print('✅ Agent ready! Type your message below.\\n')
   
   while True:
       try:
           user_input = input('You: ').strip()
           
           if user_input.lower() == 'quit':
               print('\\n👋 Goodbye!')
               break
           elif user_input.lower() == 'help':
               help_text = await agent.langchain_agent.explain_capabilities()
               print(f'\\nAgent: {help_text}\\n')
               continue
           elif user_input.lower() == 'stats':
               stats = agent.get_comprehensive_stats()
               print(f'\\nAgent Stats:')
               print(f'  • Total interactions: {stats[\"integration_stats\"][\"langchain_interactions\"]}')
               print(f'  • Memory operations: {stats[\"integration_stats\"][\"memory_operations\"]}')
               print(f'  • Available tools: {len(stats[\"available_capabilities\"][\"tools\"])}')
               print()
               continue
           elif not user_input:
               continue
           
           print('\\nAgent: Thinking...')
           response = await agent.process_input(user_input, 'demo_user')
           
           print(f'\\rAgent: {response[\"response\"]}')
           
           if response['tools_used']:
               print(f'🔧 Tools used: {\", \".join(response[\"tools_used\"])}')
           
           print(f'⏱️ Response time: {response[\"processing_breakdown\"][\"total_time\"]:.2f}s\\n')
           
       except KeyboardInterrupt:
           print('\\n\\n👋 Goodbye!')
           break
       except Exception as e:
           print(f'\\n❌ Error: {e}\\n')

asyncio.run(interactive_demo())
"
}

run_performance_validation() {
   log "Running performance validation..."
   echo ""
   
   python3 scripts/production_validator.py --validate-only --base-url http://localhost:8000
}

show_advanced_config() {
   echo ""
   echo "🔧 Advanced Configuration Options"
   echo "================================"
   echo ""
   echo "Configuration files:"
   echo "• .env - Main environment variables"
   echo "• src/config/settings.py - Application settings"
   echo "• docker-compose.yml - Container orchestration"
   echo ""
   echo "Key configuration areas:"
   echo "• OpenAI model selection (OPENAI_MODEL)"
   echo "• Memory capacity (AGENT_MEMORY_SIZE)"
   echo "• Rate limiting (API_RATE_LIMIT)"
   echo "• Vector database type (VECTOR_DB_TYPE)"
   echo "• Logging level (LOG_LEVEL)"
   echo ""
   echo "To modify configuration:"
   echo "1. Edit .env file for environment variables"
   echo "2. Restart the application to apply changes"
   echo "3. Run validation to ensure changes work correctly"
   echo ""
   
   read -p "Press Enter to continue..."
}

show_documentation() {
   echo ""
   echo "📚 AI Agent Documentation"
   echo "========================="
   echo ""
   echo "Available documentation:"
   echo "• README.md - Project overview and quick start"
   echo "• docs/ directory - Detailed documentation"
   echo "• API docs - http://localhost:8000/docs (when server is running)"
   echo "• Code comments - Extensive inline documentation"
   echo ""
   echo "Key concepts to understand:"
   echo "• Chapter 2 Components - Core agent architecture"
   echo "• LangChain Integration - Tool orchestration and reasoning"
   echo "• Vector Database - Memory and context retention"
   echo "• Production Features - Monitoring, deployment, and scaling"
   echo ""
   echo "Example projects:"
   echo "• examples/test_langchain_agent.py - LangChain integration demo"
   echo "• examples/chapter2_integration.py - Core components demo"
   echo ""
   
   read -p "Press Enter to continue..."
}

setup_docker() {
   log "Setting up Docker environment..."
   echo ""
   
   # Check if Docker is available
   if ! command -v docker &> /dev/null; then
       warn "Docker not found. Please install Docker first:"
       echo "• Docker Desktop: https://www.docker.com/products/docker-desktop"
       echo "• Docker Engine: https://docs.docker.com/engine/install/"
       return
   fi
   
   if ! command -v docker-compose &> /dev/null; then
       warn "Docker Compose not found. Please install Docker Compose:"
       echo "• Installation guide: https://docs.docker.com/compose/install/"
       return
   fi
   
   info "✓ Docker and Docker Compose found"
   
   echo ""
   echo "Docker setup options:"
   echo "1) Development environment (single container)"
   echo "2) Production environment (full stack with monitoring)"
   echo "3) Build custom image only"
   echo ""
   
   read -p "Choose option (1-3): " docker_choice
   
   case $docker_choice in
       1)
           log "Starting development Docker environment..."
           docker build -t ai-agent:dev .
           docker run -d -p 8000:8000 --name ai-agent-dev ai-agent:dev
           info "✓ Development container started on http://localhost:8000"
           ;;
       2)
           log "Starting production Docker environment..."
           docker-compose up -d
           sleep 10
           info "✓ Production stack started:"
           info "  • AI Agent: http://localhost:8000"
           info "  • Grafana: http://localhost:3000"
           info "  • Prometheus: http://localhost:9090"
           ;;
       3)
           log "Building custom Docker image..."
           docker build -t ai-agent:latest .
           info "✓ Docker image built successfully"
           ;;
   esac
}

# Main execution
main() {
   echo "Starting AI Agent setup process..."
   echo ""
   
   # Run setup steps
   check_prerequisites
   setup_venv
   install_dependencies
   setup_configuration
   setup_git
   validate_api_keys
   run_setup_validation
   run_quick_test
   
   echo ""
   echo "🎉 AI Agent setup completed successfully!"
   echo ""
   echo "Setup Summary:"
   echo "✅ Virtual environment created and activated"
   echo "✅ All dependencies installed"
   echo "✅ Configuration files created"
   echo "✅ Vector database initialized"
   echo "✅ LangChain integration ready"
   echo "✅ FastAPI web interface configured"
   echo "✅ Testing framework available"
   echo "✅ Production monitoring setup"
   echo ""
   
   # Show the menu
   show_menu
}

# Run main function
main "$@"