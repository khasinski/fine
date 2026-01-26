# frozen_string_literal: true

module Fine
  # Base error class for all Fine errors
  class Error < StandardError; end

  # Raised when a model cannot be found on Hugging Face Hub
  class ModelNotFoundError < Error
    attr_reader :model_id

    def initialize(model_id, message = nil)
      @model_id = model_id
      super(message || "Model not found: #{model_id}")
    end
  end

  # Raised when configuration is invalid
  class ConfigurationError < Error; end

  # Raised when there's an issue with dataset loading or processing
  class DatasetError < Error; end

  # Raised when training fails
  class TrainingError < Error; end

  # Raised when model weights cannot be loaded
  class WeightLoadingError < Error
    attr_reader :missing_keys, :unexpected_keys

    def initialize(message, missing_keys: [], unexpected_keys: [])
      @missing_keys = missing_keys
      @unexpected_keys = unexpected_keys
      super(message)
    end
  end

  # Raised when image processing fails
  class ImageProcessingError < Error
    attr_reader :path

    def initialize(path, message = nil)
      @path = path
      super(message || "Failed to process image: #{path}")
    end
  end

  # Raised when model export fails
  class ExportError < Error; end
end
