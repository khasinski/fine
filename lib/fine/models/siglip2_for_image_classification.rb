# frozen_string_literal: true

module Fine
  module Models
    # SigLIP2 model with classification head for image classification
    class SigLIP2ForImageClassification < Base
      attr_reader :encoder, :classifier, :num_labels

      # Load a pretrained model from Hugging Face Hub
      #
      # @param model_id [String] Hugging Face model ID (e.g., "google/siglip2-base-patch16-224")
      # @param num_labels [Integer] Number of classification labels
      # @param freeze_encoder [Boolean] Whether to freeze the vision encoder
      # @param dropout [Float] Dropout rate for classifier
      # @return [SigLIP2ForImageClassification]
      def self.from_pretrained(model_id, num_labels:, freeze_encoder: false, dropout: 0.0)
        # Download model files
        downloader = Hub::ModelDownloader.new(model_id)
        model_path = downloader.download

        # Load config
        config = Hub::ConfigLoader.from_pretrained(model_path)

        # Create model
        model = new(config, num_labels: num_labels, dropout: dropout)

        # Load pretrained weights into encoder
        weights_path = downloader.file_path("model.safetensors")
        load_result = Hub::SafetensorsLoader.load_into_model(
          model.encoder,
          weights_path,
          strict: false,
          prefix: "vision_model"
        )

        # Log any issues
        if load_result[:missing_keys].any?
          warn "Missing keys in encoder: #{load_result[:missing_keys].first(5).join(', ')}..."
        end

        # Freeze encoder if requested
        model.encoder.freeze! if freeze_encoder

        model
      end

      # Load a fine-tuned model from disk
      #
      # @param path [String] Path to saved model directory
      # @return [SigLIP2ForImageClassification]
      def self.load(path)
        raise ModelNotFoundError.new(path, "Model directory not found") unless File.directory?(path)

        # Load config
        config = Hub::ConfigLoader.from_pretrained(path)
        num_labels = config.config["num_labels"] || config.config["id2label"]&.size

        raise ConfigurationError, "Cannot determine num_labels from saved model" unless num_labels

        # Create model
        model = new(config, num_labels: num_labels)

        # Load weights
        weights_path = File.join(path, "model.safetensors")
        Hub::SafetensorsLoader.load_into_model(model, weights_path, strict: false)

        model
      end

      def initialize(config, num_labels:, dropout: 0.0)
        super(config)

        @num_labels = num_labels

        # Vision encoder
        @encoder = SigLIP2VisionEncoder.new(config)

        # Classification head
        @classifier = ClassificationHead.new(
          config.hidden_size,
          num_labels,
          dropout: dropout
        )
      end

      def forward(pixel_values, labels: nil)
        # Get image features from encoder
        features = @encoder.call(pixel_values)

        # Classification logits
        logits = @classifier.call(features)

        # Compute loss if labels provided
        if labels
          loss = Torch::NN::Functional.cross_entropy(logits, labels)
          { loss: loss, logits: logits }
        else
          { logits: logits }
        end
      end

      # Predict class for an image tensor
      #
      # @param pixel_values [Torch::Tensor] Image tensor (batch, C, H, W)
      # @return [Torch::Tensor] Predicted class indices
      def predict(pixel_values)
        eval
        Torch.no_grad do
          output = forward(pixel_values)
          output[:logits].argmax(dim: 1)
        end
      end

      # Get probabilities for each class
      #
      # @param pixel_values [Torch::Tensor] Image tensor (batch, C, H, W)
      # @return [Torch::Tensor] Class probabilities
      def predict_proba(pixel_values)
        eval
        Torch.no_grad do
          output = forward(pixel_values)
          Torch::NN::Functional.softmax(output[:logits], dim: 1)
        end
      end

      # Save the model to disk
      #
      # @param path [String] Directory path to save to
      # @param label_map [Hash, nil] Optional mapping of label names to IDs
      def save(path, label_map: nil)
        FileUtils.mkdir_p(path)

        # Save weights
        weights_path = File.join(path, "model.safetensors")
        Safetensors::Torch.save_file(state_dict, weights_path)

        # Build config with num_labels
        save_config = @config.to_h.merge(
          "num_labels" => @num_labels,
          "model_type" => "siglip2_image_classification"
        )

        # Add label mapping if provided
        if label_map
          save_config["id2label"] = label_map.invert.sort.to_h
          save_config["label2id"] = label_map
        end

        # Save config
        config_path = File.join(path, "config.json")
        File.write(config_path, JSON.pretty_generate(save_config))
      end
    end
  end
end
