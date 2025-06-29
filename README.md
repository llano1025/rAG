# rAG - Retrieval-Augmented Generation System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-75%25%20complete-orange.svg)

A production-ready **Retrieval-Augmented Generation (RAG)** system built with FastAPI for intelligent document processing, vector search, and AI-powered question answering. The system combines advanced document ingestion, vector embeddings, and multiple LLM providers to deliver accurate, context-aware responses.

## 🚀 Features

### ✅ **Implemented Features**

#### **Document Processing**
- 📄 **Multi-format Support**: PDF, DOCX, TXT, HTML, and images
- 🔍 **OCR Processing**: Tesseract-based text extraction from scanned documents
- 📊 **Metadata Extraction**: Automatic document metadata parsing
- 🔀 **Batch Upload**: Efficient processing of multiple documents
- 📁 **File Type Detection**: Automatic format recognition

#### **Vector Database & Search**
- 🧩 **Adaptive Chunking**: Context-aware text segmentation
- 🎯 **Dual Embedding Strategy**: Separate content and context embeddings
- ⚡ **FAISS Integration**: High-performance similarity search
- 🔄 **Document Versioning**: Track document changes over time
- 🎨 **Search Optimization**: Advanced ranking and filtering

#### **LLM Integration**
- 🤖 **Multi-Provider Support**: OpenAI, Ollama, LM Studio
- 📝 **Template System**: YAML-based prompt management (7 templates)
- 🔄 **Fallback Mechanisms**: Automatic provider switching
- 📡 **Streaming Support**: Real-time response generation
- 🎛️ **Context Optimization**: Intelligent context window management

#### **Security & Performance**
- 🔐 **OAuth2 Authentication**: JWT-based security
- 🛡️ **Encryption**: Fernet-based data encryption
- 📊 **Audit Logging**: Comprehensive activity tracking
- 🕵️ **PII Detection**: Privacy protection capabilities
- ⚡ **Redis Caching**: High-performance data caching
- 📈 **Health Monitoring**: System health checks and metrics

#### **API & Architecture**
- 🌐 **RESTful API**: Comprehensive endpoint coverage
- 📚 **Auto-Documentation**: OpenAPI/Swagger integration
- 🔄 **Async Architecture**: High-performance async/await patterns
- 🧩 **Modular Design**: Clean separation of concerns
- 🐳 **Docker Ready**: Multi-stage containerization

### 🚧 **In Development**
- 👥 **User Management**: Registration and RBAC endpoints
- 📊 **Database Integration**: PostgreSQL models and ORM
- 🎯 **Vector DB Client**: Qdrant integration completion
- 📝 **Document CRUD**: Complete document management
- 🧪 **Test Suite**: Comprehensive test coverage

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
├─────────────────────────────────────────────────────────────┤
│ Routes → Middleware → Controllers → Business Logic          │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layers                        │
├─────────────────────────────────────────────────────────────┤
│ • File Processing (OCR, Text Extraction, Metadata)         │
│ • Vector Processing (Chunking, Embeddings, Search)         │
│ • LLM Integration (Models, Prompts, Context)               │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Utility Services                        │
├─────────────────────────────────────────────────────────────┤
│ • Caching (Redis)           • Security (Encryption, Auth)  │
│ • Monitoring (Health, Metrics) • Export (Data, Webhooks)   │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

**Backend Framework:**
- **FastAPI** - High-performance async web framework
- **Python 3.8+** - Modern Python with type hints

**Document Processing:**
- **PyMuPDF** - PDF text extraction
- **python-docx** - Word document processing
- **Tesseract** - OCR for images and scanned documents
- **BeautifulSoup** - HTML content parsing

**AI & Vector Processing:**
- **transformers** - Hugging Face model integration
- **sentence-transformers** - Embedding generation
- **FAISS** - Vector similarity search
- **OpenAI/Ollama** - LLM providers

**Data & Caching:**
- **PostgreSQL** - Primary database (planned)
- **Qdrant** - Vector database (integration in progress)
- **Redis** - Caching and session storage

**Security & Monitoring:**
- **cryptography** - Data encryption (Fernet)
- **python-jose** - JWT token handling
- **passlib** - Password hashing

## 📋 Prerequisites

- **Python 3.8+**
- **Redis** (for caching)
- **PostgreSQL** (for data persistence)
- **Qdrant** (for vector storage)
- **Tesseract** (for OCR functionality)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd rAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Services
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or start manually
uvicorn main:app --reload
```

### 5. Access API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:

```bash
# Application
APP_NAME=rAG
APP_VERSION=1.0.0
ENVIRONMENT=development

# Security
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost/rag_db

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://localhost:6379

# LLM Providers
OPENAI_API_KEY=your-openai-key
OLLAMA_BASE_URL=http://localhost:11434
```

## 📚 API Usage

### Document Upload
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

### Search Documents
```bash
curl -X POST "http://localhost:8000/api/v1/search/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

### Generate Answer
```bash
curl -X POST "http://localhost:8000/api/v1/search/answer" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "Explain the key concepts", "template": "analytical_rag"}'
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker build -t rag-system --target production .
docker run -p 8000:8000 --env-file .env rag-system
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## 📁 Project Structure

```
rAG/
├── api/                    # FastAPI routes and middleware
│   ├── routes/            # API endpoint definitions
│   ├── controllers/       # Business logic controllers
│   ├── middleware/        # Custom middleware
│   └── schemas/           # Pydantic models
├── file_processor/        # Document processing
│   ├── text_extractor.py # Multi-format text extraction
│   ├── ocr_processor.py   # OCR functionality
│   └── metadata_extractor.py # Document metadata
├── llm/                   # LLM integration
│   ├── model_manager.py   # Multi-provider LLM manager
│   ├── templates/         # YAML prompt templates
│   └── context_optimizer.py # Context management
├── vector_db/             # Vector database operations
│   ├── chunking.py        # Adaptive text chunking
│   ├── embedding_manager.py # Embedding generation
│   └── search_optimizer.py # Search algorithms
├── utils/                 # Utility services
│   ├── caching.py         # Redis caching
│   ├── security.py        # Encryption and auth
│   ├── monitoring.py      # Health checks
│   └── export.py          # Data export
├── tests/                 # Test suite
├── main.py               # Application entry point
├── config.py             # Configuration management
└── requirements.txt      # Dependencies
```

## 🔧 Development

### Adding New LLM Providers
1. Extend `ModelManager` in `llm/model_manager.py`
2. Add provider configuration in `config.py`
3. Update environment variables

### Creating Custom Prompt Templates
1. Add templates to `llm/templates/rag_templates.yaml`
2. Follow existing template structure
3. Test with different query types

### Extending Document Processing
1. Add new extractors to `file_processor/`
2. Register in `text_extractor.py`
3. Update supported file types

## 📊 Monitoring

Access monitoring endpoints:
- **Health Check**: `/health`
- **Metrics**: `/metrics`
- **System Info**: `/health/system`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint when running
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

## 🗺️ Roadmap

### Immediate Priorities (v1.0)
- [ ] Complete database integration
- [ ] Finalize Qdrant vector DB setup
- [ ] Implement user management
- [ ] Add comprehensive testing

### Future Features (v2.0+)
- [ ] Web-based UI dashboard
- [ ] Advanced analytics
- [ ] Custom plugin system
- [ ] Multi-tenant support
- [ ] Real-time collaboration

---

**Built with ❤️ for intelligent document processing**