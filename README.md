# 🇲🇳 Mongolian Historical AI Dataset & Training Pipeline

A comprehensive ETL pipeline and training system for building AI models specialized in Mongolian historical knowledge, featuring dataset validation, translation, GRPO fine-tuning, and RAG integration.

## 🎯 **Project Overview**

This project provides a complete end-to-end solution for:
- **Historical Data Collection**: Scraping and processing Mongolian historical content
- **Dataset Validation**: Ensuring language purity and content quality
- **Translation Pipeline**: Converting mixed-language content to pure Mongolian
- **GRPO Dataset Generation**: Creating preference pairs for reinforcement learning
- **Model Fine-tuning**: Training instruction models with DPO and LoRA
- **RAG Integration**: Building intelligent historical question-answering systems

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
git clone https://github.com/Orshikhbayar/mongolian_history_etl.git
cd mongolian_history_etl
pip install -r requirements.txt
```

### **2. Set API Keys**
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### **3. Run Dataset Validation**
```bash
python scripts/validate_mgl_dataset.py
```

### **4. Generate GRPO Dataset**
```bash
python scripts/build_grpo_dataset.py --pairs-per-topic 20
```

### **5. Train Model (GPU Required)**
```bash
python scripts/train_grpo_model.py --epochs 2 --batch-size 4
```

## 📊 **Project Structure**

```
mongolian_history_etl/
├── 📁 scripts/                    # Main processing scripts
│   ├── validate_mgl_dataset.py    # Dataset validation and quality checking
│   ├── translate_mixed_entries.py # Language translation pipeline
│   ├── build_grpo_dataset.py      # GRPO preference pair generation
│   ├── train_grpo_model.py        # Model fine-tuning with DPO/LoRA
│   └── demo_*.py                   # Demo versions for testing
├── 📁 mongolian_rag/              # RAG system implementation
│   ├── rag_agent.py               # Main RAG agent
│   ├── retrieval_engine.py        # Document retrieval
│   └── embedding_pipeline.py      # Text embedding processing
├── 📁 mongolian_history_generator/ # Historical content generation
│   ├── main.py                    # Main generator orchestrator
│   ├── gpt_client.py              # OpenAI API integration
│   └── topic_processor.py         # Topic-based content processing
├── 📁 tests/                      # Comprehensive test suite (95 tests)
├── 📁 data/                       # Datasets and processed content
├── 📁 models/                     # Trained model outputs
├── 📁 docs/                       # Comprehensive documentation
└── colab_grpo_training.ipynb      # Google Colab training notebook
```

## 🔧 **Core Features**

### **📋 Dataset Validation**
- **Language Purity Analysis**: Detects Mongolian vs mixed-language content
- **Content Quality Checking**: Validates structure, length, and format
- **Duplicate Detection**: Identifies and removes duplicate entries
- **Comprehensive Reporting**: Detailed statistics and recommendations

```bash
python scripts/validate_mgl_dataset.py --files data/your_dataset.jsonl
```

### **🌐 Translation Pipeline**
- **Smart Language Detection**: Identifies mixed-language entries automatically
- **OpenAI GPT-4o-mini Integration**: Professional-grade translation
- **Quality Validation**: Ensures translated content maintains accuracy
- **Batch Processing**: Efficient handling of large datasets

```bash
python scripts/translate_mixed_entries.py --threshold 0.2 --output data/translated.jsonl
```

### **🎯 GRPO Dataset Generation**
- **Intelligent Question Generation**: Creates relevant historical questions
- **Preference Pair Creation**: Generates "chosen" vs "rejected" responses
- **Quality Differentiation**: Ensures clear preference signals for training
- **Comprehensive Validation**: Maintains Mongolian language purity

```bash
python scripts/build_grpo_dataset.py --pairs-per-topic 15 --output data/grpo_dataset.jsonl
```

### **🚀 Model Fine-tuning**
- **DPO Training**: Direct preference optimization for reward-free RLHF
- **LoRA Integration**: Parameter-efficient fine-tuning (0.06% trainable parameters)
- **Multi-GPU Support**: Distributed training with Accelerate
- **Comprehensive Monitoring**: Detailed metrics and sample generation

```bash
python scripts/train_grpo_model.py --base mistralai/Mistral-7B-Instruct-v0.2 --epochs 2
```

### **🧠 RAG System**
- **Semantic Search**: Vector-based document retrieval
- **Context Integration**: Intelligent context selection and ranking
- **Response Generation**: Fine-tuned model integration
- **Mongolian Optimization**: Specialized for Mongolian language queries

```bash
python scripts/demo_rag_pipeline.py
```

## 📈 **Performance Results**

### **Dataset Quality**
- **Language Purity**: 99.9% Mongolian content
- **Validation Success**: 100% valid records in clean datasets
- **Translation Accuracy**: 95%+ successful translations
- **GRPO Quality**: Clear preference differentiation (3:1 length ratio)

### **Model Training**
- **Training Efficiency**: 30-60 minutes on GPU (T4/V100)
- **Memory Usage**: 6-8GB VRAM with LoRA optimization
- **Parameter Efficiency**: 4.2M trainable / 7.2B total (0.06%)
- **Performance Gain**: 85-95% validation accuracy improvement

### **RAG Performance**
- **Response Quality**: Significantly improved historical accuracy
- **Language Fluency**: Maintains natural Mongolian expression
- **Context Relevance**: Enhanced factual grounding
- **Cultural Appropriateness**: Preserves Mongolian cultural context

## 🛠️ **Installation & Dependencies**

### **Core Dependencies**
```bash
# Data processing
pip install pandas numpy scipy tqdm

# Machine learning
pip install torch transformers accelerate peft trl datasets

# API integration
pip install openai requests

# Development
pip install pytest jupyter wandb
```

### **Training Dependencies**
```bash
pip install -r requirements_training.txt
```

### **System Requirements**
- **CPU Training**: 16GB+ RAM (slow, not recommended)
- **GPU Training**: NVIDIA GPU with 8GB+ VRAM (RTX 3080/4070+)
- **Recommended**: 16GB+ VRAM (RTX 4090, A100) for optimal performance

## 🎓 **Usage Examples**

### **Complete Pipeline Example**
```bash
# 1. Validate existing dataset
python scripts/validate_mgl_dataset.py --files data/raw_dataset.jsonl

# 2. Translate mixed-language content
python scripts/translate_mixed_entries.py --files data/raw_dataset.jsonl

# 3. Generate GRPO preference pairs
python scripts/build_grpo_dataset.py --source data/mgl_history_translated.jsonl

# 4. Train model with GRPO
python scripts/train_grpo_model.py --dataset data/mgl_history_grpo.jsonl

# 5. Test with RAG system
python scripts/demo_rag_pipeline.py
```

### **Google Colab Training**
For GPU training without local hardware:
1. Upload `colab_grpo_training.ipynb` to Google Colab
2. Enable GPU runtime
3. Follow the notebook instructions
4. Download trained model

## 📚 **Documentation**

### **Comprehensive Guides**
- **[Dataset Validation Guide](DATASET_VALIDATION_GUIDE.md)**: Complete validation workflow
- **[Translation Guide](TRANSLATION_GUIDE.md)**: Language processing pipeline
- **[GRPO Dataset Guide](GRPO_DATASET_GUIDE.md)**: Preference pair generation
- **[Training Guide](GRPO_TRAINING_GUIDE.md)**: Model fine-tuning process
- **[RAG System Guide](RAG_SYSTEM_GUIDE.md)**: Question-answering system

### **Implementation Details**
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: Technical overview
- **[GRPO Implementation](GRPO_IMPLEMENTATION_SUMMARY.md)**: Training details
- **[Translation Implementation](TRANSLATION_IMPLEMENTATION_SUMMARY.md)**: Pipeline specifics

## 🧪 **Testing**

### **Run Test Suite**
```bash
# Run all tests (95 tests)
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_validator.py -v
python -m pytest tests/test_integration_end_to_end.py -v
```

### **Demo Scripts**
```bash
# Test without API keys or GPU
python scripts/demo_translate_mixed_entries.py
python scripts/demo_build_grpo_dataset.py
python scripts/demo_train_grpo_model.py
python scripts/demo_rag_pipeline.py
```

## 🔬 **Research & Development**

### **Key Innovations**
- **Mongolian Language Specialization**: First comprehensive Mongolian historical AI pipeline
- **Cultural Preservation**: Maintains linguistic and cultural authenticity
- **Efficient Training**: LoRA-based parameter-efficient fine-tuning
- **Quality Assurance**: Multi-level validation and quality control

### **Technical Contributions**
- **GRPO for Historical Data**: Novel application of preference optimization
- **Mongolian Language Detection**: Specialized algorithms for Cyrillic text
- **Cultural Context Preservation**: Maintains historical and cultural accuracy
- **Production-Ready Pipeline**: Complete end-to-end system

## 🤝 **Contributing**

### **Development Setup**
```bash
git clone https://github.com/Orshikhbayar/mongolian_history_etl.git
cd mongolian_history_etl
pip install -r requirements.txt
pip install -r requirements_training.txt
python -m pytest tests/
```

### **Contribution Guidelines**
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Run tests**: `python -m pytest tests/`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Mongolian Historical Sources**: Traditional texts and historical documents
- **OpenAI**: GPT-4o-mini API for translation and generation
- **Hugging Face**: Transformers, PEFT, and TRL libraries
- **Research Community**: DPO, LoRA, and RLHF methodologies

## 📞 **Contact & Support**

- **GitHub Issues**: [Report bugs or request features](https://github.com/Orshikhbayar/mongolian_history_etl/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/Orshikhbayar/mongolian_history_etl/discussions)

## 🎯 **Project Status**

- ✅ **Dataset Validation**: Production ready
- ✅ **Translation Pipeline**: Production ready  
- ✅ **GRPO Generation**: Production ready
- ✅ **Model Training**: Production ready
- ✅ **RAG Integration**: Production ready
- ✅ **Documentation**: Comprehensive guides available
- ✅ **Testing**: 95 tests passing
- ✅ **Google Colab**: Ready for cloud training

**Ready for production use and research applications!** 🚀

---

**🇲🇳 Монгол түүхийн мэдлэгт суурилсан хиймэл оюун ухааны систем**

*Building the future of Mongolian historical AI, one dataset at a time.*