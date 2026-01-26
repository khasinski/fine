# frozen_string_literal: true

module Fine
  module Models
    # Sentence Transformer for generating text embeddings
    #
    # Produces dense vector representations of sentences for semantic similarity,
    # clustering, and retrieval tasks.
    class SentenceTransformer < Base
      attr_reader :encoder, :pooling_mode

      POOLING_MODES = %i[cls mean max].freeze

      # Load a pretrained sentence transformer
      #
      # @param model_id [String] HuggingFace model ID
      # @param pooling_mode [Symbol] Pooling strategy (:cls, :mean, :max)
      # @return [SentenceTransformer]
      def self.from_pretrained(model_id, pooling_mode: :mean)
        downloader = Hub::ModelDownloader.new(model_id)
        model_path = downloader.download

        config = Hub::ConfigLoader.from_pretrained(model_path)

        model = new(config, pooling_mode: pooling_mode)

        # Load weights
        weights_path = downloader.file_path("model.safetensors")
        if File.exist?(weights_path)
          load_pretrained_weights(model, weights_path)
        end

        model
      end

      # Load from saved model
      def self.load(path)
        raise ModelNotFoundError.new(path) unless File.directory?(path)

        config = Hub::ConfigLoader.from_pretrained(path)
        pooling_mode = (config.config["pooling_mode"] || "mean").to_sym

        model = new(config, pooling_mode: pooling_mode)

        weights_path = File.join(path, "model.safetensors")
        Hub::SafetensorsLoader.load_into_model(model, weights_path, strict: false)

        model
      end

      def initialize(config, pooling_mode: :mean)
        super(config)

        raise ArgumentError, "Invalid pooling mode: #{pooling_mode}" unless POOLING_MODES.include?(pooling_mode)

        @pooling_mode = pooling_mode
        @encoder = BertEncoder.new(config)

        # Optional: normalize embeddings
        @normalize = true
      end

      def forward(input_ids, attention_mask: nil, token_type_ids: nil)
        encoder_output = @encoder.call(
          input_ids,
          attention_mask: attention_mask,
          token_type_ids: token_type_ids
        )

        # Pool based on strategy
        embeddings = case @pooling_mode
        when :cls
          encoder_output[:pooler_output]
        when :mean
          mean_pooling(encoder_output[:last_hidden_state], attention_mask)
        when :max
          max_pooling(encoder_output[:last_hidden_state], attention_mask)
        end

        # L2 normalize
        embeddings = Torch::NN::Functional.normalize(embeddings, p: 2, dim: 1) if @normalize

        { embeddings: embeddings }
      end

      # Encode texts to embeddings
      #
      # @param input_ids [Torch::Tensor] Token IDs
      # @param attention_mask [Torch::Tensor] Attention mask
      # @return [Torch::Tensor] Embeddings
      def encode(input_ids, attention_mask: nil, token_type_ids: nil)
        eval
        Torch.no_grad do
          output = forward(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids)
          output[:embeddings]
        end
      end

      # Compute similarity between two sets of embeddings
      #
      # @param embeddings_a [Torch::Tensor] First embeddings
      # @param embeddings_b [Torch::Tensor] Second embeddings
      # @return [Torch::Tensor] Similarity scores
      def similarity(embeddings_a, embeddings_b)
        # Cosine similarity (embeddings should be normalized)
        Torch.matmul(embeddings_a, embeddings_b.transpose(0, 1))
      end

      def save(path)
        FileUtils.mkdir_p(path)

        weights_path = File.join(path, "model.safetensors")
        Safetensors::Torch.save_file(state_dict, weights_path)

        save_config = @config.to_h.merge(
          "model_type" => "sentence_transformer",
          "pooling_mode" => @pooling_mode.to_s
        )

        config_path = File.join(path, "config.json")
        File.write(config_path, JSON.pretty_generate(save_config))
      end

      private

      def mean_pooling(hidden_states, attention_mask)
        if attention_mask
          mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float
          sum_embeddings = (hidden_states * mask).sum(dim: 1)
          sum_mask = mask.sum(dim: 1).clamp(min: 1e-9)
          sum_embeddings / sum_mask
        else
          hidden_states.mean(dim: 1)
        end
      end

      def max_pooling(hidden_states, attention_mask)
        if attention_mask
          mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
          hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
        end
        hidden_states.max(dim: 1).values
      end

      def self.load_pretrained_weights(model, weights_path)
        tensors = Safetensors::Torch.load_file(weights_path)

        mapped = {}
        model_keys = model.state_dict.keys

        tensors.each do |name, tensor|
          # Try direct mapping first
          mapped_name = map_sentence_transformer_weight_name(name)

          if model_keys.include?(mapped_name)
            mapped[mapped_name] = tensor
          end
        end

        # Use no_grad and copy! since torch.rb doesn't support strict: false
        Torch.no_grad do
          state_dict = model.state_dict
          mapped.each do |name, tensor|
            state_dict[name].copy!(tensor) if state_dict.key?(name)
          end
        end
      end

      def self.map_sentence_transformer_weight_name(hf_name)
        name = hf_name.dup

        # sentence-transformers uses "0." prefix for the encoder module
        name = name.sub(/^0\./, "encoder.")

        # Then apply BERT mappings
        name = name.sub("auto_model.", "encoder.")
        name = name.sub("bert.embeddings.word_embeddings", "encoder.word_embeddings")
        name = name.sub("bert.embeddings.position_embeddings", "encoder.position_embeddings")
        name = name.sub("bert.embeddings.token_type_embeddings", "encoder.token_type_embeddings")
        name = name.sub("bert.embeddings.LayerNorm", "encoder.embeddings_layer_norm")
        name = name.sub("embeddings.word_embeddings", "encoder.word_embeddings")
        name = name.sub("embeddings.position_embeddings", "encoder.position_embeddings")
        name = name.sub("embeddings.token_type_embeddings", "encoder.token_type_embeddings")
        name = name.sub("embeddings.LayerNorm", "encoder.embeddings_layer_norm")
        name = name.gsub("bert.encoder.layer", "encoder.layers")
        name = name.gsub("encoder.layer", "encoder.layers")
        name = name.gsub(".attention.self.query", ".attention.query")
        name = name.gsub(".attention.self.key", ".attention.key")
        name = name.gsub(".attention.self.value", ".attention.value")
        name = name.gsub(".attention.output.dense", ".attention.out")
        name = name.gsub(".attention.output.LayerNorm", ".attention_layer_norm")
        name = name.gsub(".intermediate.dense", ".intermediate")
        name = name.gsub(".output.dense", ".output")
        name = name.gsub(".output.LayerNorm", ".output_layer_norm")
        name = name.sub("bert.pooler.dense", "encoder.pooler_dense")
        name = name.sub("pooler.dense", "encoder.pooler_dense")

        name
      end
    end
  end
end
