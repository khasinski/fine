# frozen_string_literal: true

require "torch"
require "safetensors"
require "vips"
require "tokenizers"
require "tty-progressbar"
require "down"
require "json"
require "fileutils"

require_relative "fine/version"
require_relative "fine/error"
require_relative "fine/configuration"
require_relative "fine/validators"

# Hub
require_relative "fine/hub/config_loader"
require_relative "fine/hub/model_downloader"
require_relative "fine/hub/safetensors_loader"

# Tokenizers
require_relative "fine/tokenizers/auto_tokenizer"

# Transforms (Image)
require_relative "fine/transforms/compose"
require_relative "fine/transforms/resize"
require_relative "fine/transforms/normalize"
require_relative "fine/transforms/to_tensor"

# Datasets
require_relative "fine/datasets/image_dataset"
require_relative "fine/datasets/data_loader"
require_relative "fine/datasets/text_dataset"
require_relative "fine/datasets/text_data_loader"
require_relative "fine/datasets/instruction_dataset"

# Models - Vision
require_relative "fine/models/base"
require_relative "fine/models/siglip2_vision_encoder"
require_relative "fine/models/classification_head"
require_relative "fine/models/siglip2_for_image_classification"

# Models - Text
require_relative "fine/models/bert_encoder"
require_relative "fine/models/bert_for_sequence_classification"
require_relative "fine/models/sentence_transformer"

# Models - LLM
require_relative "fine/models/llama_decoder"
require_relative "fine/models/gemma3_decoder"
require_relative "fine/models/causal_lm"

# Training
require_relative "fine/training/trainer"
require_relative "fine/training/text_trainer"
require_relative "fine/training/llm_trainer"

# Callbacks
require_relative "fine/callbacks/base"
require_relative "fine/callbacks/progress_bar"

# High-level API
require_relative "fine/image_classifier"
require_relative "fine/text_classifier"
require_relative "fine/text_embedder"
require_relative "fine/llm"

# Export
require_relative "fine/export"

# LoRA
require_relative "fine/lora"

module Fine
  class << self
    attr_accessor :configuration

    def configure
      self.configuration ||= GlobalConfiguration.new
      yield(configuration) if block_given?
      configuration
    end

    def cache_dir
      configuration&.cache_dir || File.expand_path("~/.cache/fine")
    end

    def device
      configuration&.device || detect_device
    end

    private

    def detect_device
      if Torch::CUDA.available?
        "cuda"
      elsif defined?(Torch::Backends::MPS) && Torch::Backends::MPS.available?
        "mps"
      else
        "cpu"
      end
    end
  end

  class GlobalConfiguration
    attr_accessor :cache_dir, :device, :log_level, :progress_bar

    def initialize
      @cache_dir = File.expand_path("~/.cache/fine")
      @device = nil # auto-detect
      @log_level = :info
      @progress_bar = true
    end
  end
end
