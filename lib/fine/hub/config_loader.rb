# frozen_string_literal: true

module Fine
  module Hub
    # Loads and parses model configuration from Hugging Face format
    class ConfigLoader
      attr_reader :config

      def initialize(config_path)
        @config_path = config_path
        @config = load_config
      end

      def self.from_pretrained(model_path)
        config_path = File.join(model_path, "config.json")
        raise ConfigurationError, "Config not found: #{config_path}" unless File.exist?(config_path)

        new(config_path)
      end

      # Vision encoder configuration
      def hidden_size
        vision_config["hidden_size"] || config["hidden_size"] || 768
      end

      def num_hidden_layers
        vision_config["num_hidden_layers"] || config["num_hidden_layers"] || 12
      end

      def num_attention_heads
        vision_config["num_attention_heads"] || config["num_attention_heads"] || 12
      end

      def intermediate_size
        vision_config["intermediate_size"] || config["intermediate_size"] || 3072
      end

      def image_size
        vision_config["image_size"] || config["image_size"] || 224
      end

      def patch_size
        vision_config["patch_size"] || config["patch_size"] || 16
      end

      def num_channels
        vision_config["num_channels"] || config["num_channels"] || 3
      end

      def layer_norm_eps
        vision_config["layer_norm_eps"] || config["layer_norm_eps"] || 1e-6
      end

      def hidden_act
        vision_config["hidden_act"] || config["hidden_act"] || "gelu"
      end

      def attention_dropout
        vision_config["attention_dropout"] || config["attention_dropout"] || 0.0
      end

      # Text model configuration (BERT, DistilBERT, DeBERTa, etc.)
      def vocab_size
        config["vocab_size"] || 30522
      end

      def max_position_embeddings
        config["max_position_embeddings"] || 512
      end

      def type_vocab_size
        config["type_vocab_size"] || 2
      end

      def hidden_dropout_prob
        config["hidden_dropout_prob"] || config["dropout"] || 0.1
      end

      # LLM configuration (Llama, Gemma, etc.)
      def rope_theta
        config["rope_theta"] || 10000.0
      end

      def rms_norm_eps
        config["rms_norm_eps"] || 1e-6
      end

      def num_key_value_heads
        config["num_key_value_heads"] || num_attention_heads
      end

      def use_bias
        config["use_bias"] != false
      end

      def head_dim
        config["head_dim"] || (hidden_size / num_attention_heads)
      end

      def attention_bias
        config["attention_bias"] || false
      end

      def use_qk_norm
        # Gemma 3 uses QK normalization
        config.key?("query_pre_attn_scalar") || model_type&.include?("gemma3")
      end

      def use_pre_feedforward_layernorm
        # Gemma 3 has additional layer norms
        model_type&.include?("gemma3")
      end

      # Computed properties
      def num_patches
        (image_size / patch_size) ** 2
      end

      def model_type
        config["model_type"]
      end

      # Raw access to config sections
      def vision_config
        config["vision_config"] || {}
      end

      def text_config
        config["text_config"] || {}
      end

      def to_h
        @config
      end

      private

      def load_config
        JSON.parse(File.read(@config_path))
      rescue JSON::ParserError => e
        raise ConfigurationError, "Invalid JSON in config: #{e.message}"
      end
    end
  end
end
