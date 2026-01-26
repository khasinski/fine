# frozen_string_literal: true

require_relative "export/onnx_exporter"
require_relative "export/gguf_exporter"

module Fine
  # Export models to various deployment formats
  #
  # @example Export to ONNX
  #   Fine::Export.to_onnx(classifier, "model.onnx")
  #
  # @example Export LLM to GGUF
  #   Fine::Export.to_gguf(llm, "model.gguf", quantization: :q4_0)
  module Export
    class << self
      # Export any Fine model to ONNX format
      #
      # @param model [TextClassifier, TextEmbedder, ImageClassifier, LLM] The model
      # @param path [String] Output path
      # @param options [Hash] Export options
      # @return [String] The output path
      def to_onnx(model, path, **options)
        ONNXExporter.export(model, path, **options)
      end

      # Export LLM to GGUF format
      #
      # @param model [LLM] The LLM model
      # @param path [String] Output path
      # @param quantization [Symbol] Quantization type (:f16, :q4_0, :q8_0, etc.)
      # @param metadata [Hash] Additional metadata
      # @return [String] The output path
      def to_gguf(model, path, quantization: :f16, **options)
        GGUFExporter.export(model, path, quantization: quantization, **options)
      end

      # List available quantization options for GGUF
      #
      # @return [Hash] Quantization types with descriptions
      def gguf_quantization_options
        {
          f32: "32-bit float (largest, no quality loss)",
          f16: "16-bit float (good balance)",
          q8_0: "8-bit quantization (smaller, minimal quality loss)",
          q4_0: "4-bit quantization (smallest, some quality loss)",
          q4_k: "4-bit K-quant (better quality than q4_0)",
          q5_k: "5-bit K-quant (good quality/size balance)",
          q6_k: "6-bit K-quant (high quality)"
        }
      end
    end
  end
end
