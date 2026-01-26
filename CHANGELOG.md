# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-01-26

### Added

- **Image Classification** with SigLIP2 models
  - `Fine::ImageClassifier` for training and inference
  - Support for `google/siglip2-base-patch16-224` and other SigLIP2 variants
  - Directory-based dataset loading with automatic class discovery

- **Text Classification** with BERT-family models
  - `Fine::TextClassifier` for sentiment, intent, and category classification
  - Support for DistilBERT, BERT, DeBERTa models
  - JSONL dataset format with `text` and `label` fields

- **Text Embeddings** with Sentence Transformers
  - `Fine::TextEmbedder` for domain-specific embeddings
  - Support for MiniLM, MPNet, BGE models
  - Contrastive learning with positive/negative pairs

- **LLM Fine-tuning** (experimental)
  - `Fine::LLM` for instruction fine-tuning
  - Support for Llama, Gemma, Qwen architectures
  - Alpaca-format dataset support

- **Model Export**
  - ONNX export for all model types
  - GGUF export for LLMs (llama.cpp/Ollama compatible)

- **Infrastructure**
  - HuggingFace Hub integration with automatic model downloads
  - SafeTensors weight loading
  - Block-based configuration DSL
  - Progress bar callbacks
  - MPS (Apple Silicon) and CUDA support
