# frozen_string_literal: true

module Fine
  module Export
    # Export models to ONNX format
    #
    # @example Export a text classifier
    #   classifier = Fine::TextClassifier.load("my_model")
    #   Fine::Export::ONNXExporter.export(classifier, "model.onnx")
    #
    # @example Export with options
    #   Fine::Export::ONNXExporter.export(
    #     model,
    #     "model.onnx",
    #     opset_version: 14,
    #     dynamic_axes: true
    #   )
    class ONNXExporter
      SUPPORTED_TYPES = [
        Fine::TextClassifier,
        Fine::TextEmbedder,
        Fine::ImageClassifier,
        Fine::LLM
      ].freeze

      class << self
        # Export a Fine model to ONNX format
        #
        # @param fine_model [TextClassifier, TextEmbedder, ImageClassifier, LLM] The model to export
        # @param output_path [String] Path for the output ONNX file
        # @param opset_version [Integer] ONNX opset version (default: 14)
        # @param dynamic_axes [Boolean] Use dynamic axes for variable batch/sequence (default: true)
        # @param quantize [Symbol, nil] Quantization type (:int8, :uint8, nil)
        def export(fine_model, output_path, opset_version: 14, dynamic_axes: true, quantize: nil)
          validate_model(fine_model)

          model = fine_model.model
          model.eval

          # Get example inputs based on model type
          example_inputs, input_names, output_names, dynamic_axes_config =
            prepare_export_config(fine_model, dynamic_axes)

          # Export to ONNX
          Torch::ONNX.export(
            model,
            example_inputs,
            output_path,
            input_names: input_names,
            output_names: output_names,
            dynamic_axes: dynamic_axes_config,
            opset_version: opset_version,
            do_constant_folding: true
          )

          # Optional quantization
          if quantize
            quantize_model(output_path, quantize)
          end

          output_path
        end

        # Export only the encoder/backbone (useful for embeddings)
        #
        # @param fine_model [TextEmbedder, ImageClassifier] Model with encoder
        # @param output_path [String] Output path
        def export_encoder(fine_model, output_path, **options)
          unless fine_model.respond_to?(:model) && fine_model.model.respond_to?(:encoder)
            raise ExportError, "Model does not have an encoder"
          end

          encoder = fine_model.model.encoder
          encoder.eval

          example_inputs, input_names, output_names, dynamic_axes_config =
            prepare_encoder_config(fine_model)

          Torch::ONNX.export(
            encoder,
            example_inputs,
            output_path,
            input_names: input_names,
            output_names: output_names,
            dynamic_axes: dynamic_axes_config,
            opset_version: options[:opset_version] || 14
          )

          output_path
        end

        private

        def validate_model(model)
          unless SUPPORTED_TYPES.any? { |t| model.is_a?(t) }
            raise ExportError, "Unsupported model type: #{model.class}"
          end

          unless model.model
            raise ExportError, "Model not loaded or trained"
          end
        end

        def prepare_export_config(fine_model, dynamic_axes)
          case fine_model
          when Fine::TextClassifier, Fine::TextEmbedder
            prepare_text_config(fine_model, dynamic_axes)
          when Fine::ImageClassifier
            prepare_image_config(fine_model, dynamic_axes)
          when Fine::LLM
            prepare_llm_config(fine_model, dynamic_axes)
          end
        end

        def prepare_text_config(fine_model, dynamic_axes)
          batch_size = 1
          seq_length = fine_model.config.max_length

          example_inputs = [
            Torch.zeros([batch_size, seq_length], dtype: :int64),  # input_ids
            Torch.ones([batch_size, seq_length], dtype: :int64)    # attention_mask
          ]

          input_names = %w[input_ids attention_mask]

          output_names = if fine_model.is_a?(Fine::TextEmbedder)
            %w[embeddings]
          else
            %w[logits]
          end

          dynamic_axes_config = if dynamic_axes
            {
              "input_ids" => { 0 => "batch_size", 1 => "sequence" },
              "attention_mask" => { 0 => "batch_size", 1 => "sequence" },
              output_names.first => { 0 => "batch_size" }
            }
          end

          [example_inputs, input_names, output_names, dynamic_axes_config]
        end

        def prepare_image_config(fine_model, dynamic_axes)
          # Get image size from config
          image_size = fine_model.config.image_size || 224
          batch_size = 1

          example_inputs = [
            Torch.zeros([batch_size, 3, image_size, image_size], dtype: :float32)
          ]

          input_names = %w[pixel_values]
          output_names = %w[logits]

          dynamic_axes_config = if dynamic_axes
            {
              "pixel_values" => { 0 => "batch_size" },
              "logits" => { 0 => "batch_size" }
            }
          end

          [example_inputs, input_names, output_names, dynamic_axes_config]
        end

        def prepare_llm_config(fine_model, dynamic_axes)
          batch_size = 1
          seq_length = 128  # Smaller default for export

          example_inputs = [
            Torch.zeros([batch_size, seq_length], dtype: :int64)  # input_ids
          ]

          input_names = %w[input_ids]
          output_names = %w[logits]

          dynamic_axes_config = if dynamic_axes
            {
              "input_ids" => { 0 => "batch_size", 1 => "sequence" },
              "logits" => { 0 => "batch_size", 1 => "sequence" }
            }
          end

          [example_inputs, input_names, output_names, dynamic_axes_config]
        end

        def prepare_encoder_config(fine_model)
          case fine_model
          when Fine::TextEmbedder
            batch_size = 1
            seq_length = fine_model.config.max_length

            example_inputs = [
              Torch.zeros([batch_size, seq_length], dtype: :int64),
              Torch.ones([batch_size, seq_length], dtype: :int64)
            ]

            input_names = %w[input_ids attention_mask]
            output_names = %w[last_hidden_state]

            dynamic_axes_config = {
              "input_ids" => { 0 => "batch_size", 1 => "sequence" },
              "attention_mask" => { 0 => "batch_size", 1 => "sequence" },
              "last_hidden_state" => { 0 => "batch_size", 1 => "sequence" }
            }

            [example_inputs, input_names, output_names, dynamic_axes_config]
          when Fine::ImageClassifier
            image_size = fine_model.config.image_size || 224

            example_inputs = [
              Torch.zeros([1, 3, image_size, image_size], dtype: :float32)
            ]

            [example_inputs, %w[pixel_values], %w[features], { "pixel_values" => { 0 => "batch_size" } }]
          end
        end

        def quantize_model(model_path, quantize_type)
          # Note: Full ONNX quantization requires onnxruntime
          # This is a placeholder for the quantization logic
          require "onnxruntime"

          quantized_path = model_path.sub(".onnx", "_quantized.onnx")

          case quantize_type
          when :int8
            # Dynamic INT8 quantization
            OnnxRuntime::Quantization.quantize_dynamic(
              model_path,
              quantized_path,
              weight_type: :int8
            )
          when :uint8
            OnnxRuntime::Quantization.quantize_dynamic(
              model_path,
              quantized_path,
              weight_type: :uint8
            )
          end

          # Replace original with quantized
          FileUtils.mv(quantized_path, model_path)
        rescue LoadError
          warn "onnxruntime gem not installed, skipping quantization"
        end
      end
    end
  end
end
