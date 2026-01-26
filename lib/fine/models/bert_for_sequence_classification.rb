# frozen_string_literal: true

module Fine
  module Models
    # BERT model with classification head for sequence classification
    class BertForSequenceClassification < Base
      attr_reader :encoder, :classifier, :num_labels

      # Load a pretrained model from Hugging Face Hub
      #
      # @param model_id [String] HuggingFace model ID
      # @param num_labels [Integer] Number of classification labels
      # @param dropout [Float] Dropout rate for classifier
      # @return [BertForSequenceClassification]
      def self.from_pretrained(model_id, num_labels:, dropout: 0.1)
        # Download model files
        downloader = Hub::ModelDownloader.new(model_id)
        model_path = downloader.download

        # Load config
        config = Hub::ConfigLoader.from_pretrained(model_path)

        # Create model
        model = new(config, num_labels: num_labels, dropout: dropout)

        # Load pretrained weights into encoder
        weights_path = downloader.file_path("model.safetensors")

        if File.exist?(weights_path)
          load_result = load_pretrained_weights(model, weights_path)

          if load_result[:missing_keys].any?
            # Only warn about unexpected missing keys (classifier is expected to be missing)
            encoder_missing = load_result[:missing_keys].reject { |k| k.include?("classifier") }
            if encoder_missing.any?
              warn "Missing encoder keys: #{encoder_missing.first(5).join(', ')}..."
            end
          end
        end

        model
      end

      # Load from saved fine-tuned model
      #
      # @param path [String] Path to saved model directory
      # @return [BertForSequenceClassification]
      def self.load(path)
        raise ModelNotFoundError.new(path) unless File.directory?(path)

        config = Hub::ConfigLoader.from_pretrained(path)
        num_labels = config.config["num_labels"] || config.config["id2label"]&.size

        raise ConfigurationError, "Cannot determine num_labels from saved model" unless num_labels

        model = new(config, num_labels: num_labels)

        weights_path = File.join(path, "model.safetensors")
        Hub::SafetensorsLoader.load_into_model(model, weights_path, strict: false)

        model
      end

      def initialize(config, num_labels:, dropout: 0.1)
        super(config)

        @num_labels = num_labels

        # Encoder
        @encoder = BertEncoder.new(config)

        # Classification head
        @dropout = Torch::NN::Dropout.new(p: dropout)
        @classifier = Torch::NN::Linear.new(config.hidden_size, num_labels)
      end

      def forward(input_ids, attention_mask: nil, token_type_ids: nil, labels: nil)
        # Get encoder outputs
        encoder_output = @encoder.call(
          input_ids,
          attention_mask: attention_mask,
          token_type_ids: token_type_ids
        )

        # Use pooled output for classification
        pooled_output = encoder_output[:pooler_output]
        pooled_output = @dropout.call(pooled_output)

        # Classification logits
        logits = @classifier.call(pooled_output)

        # Compute loss if labels provided
        if labels
          loss = Torch::NN::Functional.cross_entropy(logits, labels)
          { loss: loss, logits: logits }
        else
          { logits: logits }
        end
      end

      # Predict class for input
      #
      # @param input_ids [Torch::Tensor] Input token IDs
      # @param attention_mask [Torch::Tensor] Attention mask
      # @return [Torch::Tensor] Predicted class indices
      def predict(input_ids, attention_mask: nil, token_type_ids: nil)
        eval
        Torch.no_grad do
          output = forward(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids)
          output[:logits].argmax(dim: 1)
        end
      end

      # Get class probabilities
      #
      # @return [Torch::Tensor] Class probabilities
      def predict_proba(input_ids, attention_mask: nil, token_type_ids: nil)
        eval
        Torch.no_grad do
          output = forward(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids)
          Torch::NN::Functional.softmax(output[:logits], dim: 1)
        end
      end

      # Save the model
      #
      # @param path [String] Directory path
      # @param label_map [Hash, nil] Label mapping
      def save(path, label_map: nil)
        FileUtils.mkdir_p(path)

        # Save weights
        weights_path = File.join(path, "model.safetensors")
        Safetensors::Torch.save_file(state_dict, weights_path)

        # Build config
        save_config = @config.to_h.merge(
          "num_labels" => @num_labels,
          "model_type" => "bert_classification"
        )

        if label_map
          save_config["id2label"] = label_map.invert.sort.to_h
          save_config["label2id"] = label_map
        end

        config_path = File.join(path, "config.json")
        File.write(config_path, JSON.pretty_generate(save_config))
      end

      private

      def self.load_pretrained_weights(model, weights_path)
        tensors = Safetensors::Torch.load_file(weights_path)

        # Map HuggingFace weight names to our model structure
        mapped = {}
        missing_keys = []
        unexpected_keys = []

        model_keys = model.state_dict.keys

        tensors.each do |name, tensor|
          mapped_name = map_bert_weight_name(name)

          if model_keys.include?(mapped_name)
            mapped[mapped_name] = tensor
          else
            unexpected_keys << name
          end
        end

        missing_keys = model_keys - mapped.keys

        # Use no_grad and copy! since torch.rb doesn't support strict: false
        Torch.no_grad do
          state_dict = model.state_dict
          mapped.each do |name, tensor|
            state_dict[name].copy!(tensor) if state_dict.key?(name)
          end
        end

        { missing_keys: missing_keys, unexpected_keys: unexpected_keys }
      end

      def self.map_bert_weight_name(hf_name)
        name = hf_name.dup

        # Embeddings
        name = name.sub("bert.embeddings.word_embeddings", "encoder.word_embeddings")
        name = name.sub("bert.embeddings.position_embeddings", "encoder.position_embeddings")
        name = name.sub("bert.embeddings.token_type_embeddings", "encoder.token_type_embeddings")
        name = name.sub("bert.embeddings.LayerNorm", "encoder.embeddings_layer_norm")
        name = name.sub("embeddings.word_embeddings", "encoder.word_embeddings")
        name = name.sub("embeddings.position_embeddings", "encoder.position_embeddings")
        name = name.sub("embeddings.token_type_embeddings", "encoder.token_type_embeddings")
        name = name.sub("embeddings.LayerNorm", "encoder.embeddings_layer_norm")

        # Encoder layers
        name = name.gsub("bert.encoder.layer", "encoder.layers")
        name = name.gsub("encoder.layer", "encoder.layers")

        # Attention
        name = name.gsub(".attention.self.query", ".attention.query")
        name = name.gsub(".attention.self.key", ".attention.key")
        name = name.gsub(".attention.self.value", ".attention.value")
        name = name.gsub(".attention.output.dense", ".attention.out")
        name = name.gsub(".attention.output.LayerNorm", ".attention_layer_norm")

        # FFN
        name = name.gsub(".intermediate.dense", ".intermediate")
        name = name.gsub(".output.dense", ".output")
        name = name.gsub(".output.LayerNorm", ".output_layer_norm")

        # Pooler
        name = name.sub("bert.pooler.dense", "encoder.pooler_dense")
        name = name.sub("pooler.dense", "encoder.pooler_dense")

        # Classifier
        name = name.sub("classifier", "classifier")

        name
      end
    end
  end
end
