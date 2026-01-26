# frozen_string_literal: true

module Fine
  module Models
    # Causal Language Model for text generation
    #
    # Wraps a decoder model with a language modeling head for next-token prediction.
    class CausalLM < Base
      attr_reader :decoder, :lm_head

      # Load from HuggingFace Hub
      #
      # @param model_id [String] Model ID (e.g., "google/gemma-2b", "meta-llama/Llama-3.2-1B")
      # @param dtype [Symbol] Data type (:float32, :float16, :bfloat16, or :auto)
      # @return [CausalLM]
      def self.from_pretrained(model_id, dtype: :auto)
        downloader = Hub::ModelDownloader.new(model_id)
        model_path = downloader.download

        config = Hub::ConfigLoader.from_pretrained(model_path)

        # Determine dtype from config if auto
        if dtype == :auto
          config_dtype = config.config["torch_dtype"]
          dtype = case config_dtype
                  when "bfloat16" then :bfloat16
                  when "float16" then :float16
                  else :float32
                  end
        end

        model = new(config)

        # Convert model to target dtype before loading weights (saves memory)
        model.to(dtype) if dtype != :float32

        # Load weights
        weights_path = downloader.file_path("model.safetensors")
        if File.exist?(weights_path)
          load_pretrained_weights(model, weights_path)
        else
          # Try sharded weights
          load_sharded_weights(model, model_path)
        end

        model
      end

      # Load from saved model
      def self.load(path)
        raise ModelNotFoundError.new(path) unless File.directory?(path)

        config = Hub::ConfigLoader.from_pretrained(path)
        model = new(config)

        weights_path = File.join(path, "model.safetensors")
        Hub::SafetensorsLoader.load_into_model(model, weights_path, strict: false)

        model
      end

      def initialize(config)
        super(config)

        # Use appropriate decoder based on model type
        @decoder = if config.model_type&.include?("gemma3")
          Gemma3Decoder.new(config)
        else
          LlamaDecoder.new(config)
        end

        # LM head (often tied to embeddings)
        @lm_head = Torch::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)

        # Optionally tie weights with embeddings
        @tie_word_embeddings = config.config["tie_word_embeddings"] != false
      end

      def forward(input_ids, attention_mask: nil, labels: nil, return_logits: nil)
        # Default: return logits only if no labels (inference mode)
        return_logits = labels.nil? if return_logits.nil?

        # Get decoder outputs
        outputs = @decoder.call(input_ids, attention_mask: attention_mask)
        hidden_states = outputs[:last_hidden_state]

        # LM head
        logits = @lm_head.call(hidden_states)

        # Compute loss if labels provided
        if labels
          # Shift for next-token prediction
          shift_logits = logits[0.., 0...-1, 0..].contiguous
          shift_labels = labels[0.., 1..].contiguous

          # Compute cross entropy loss
          vocab_size = logits.size(-1)
          loss = Torch::NN::Functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index: -100
          )

          # Don't return logits during training to save memory
          return_logits ? { loss: loss, logits: logits } : { loss: loss }
        else
          { logits: logits }
        end
      end

      # Generate text autoregressively
      #
      # @param input_ids [Torch::Tensor] Input token IDs
      # @param max_new_tokens [Integer] Maximum tokens to generate
      # @param temperature [Float] Sampling temperature
      # @param top_p [Float] Nucleus sampling threshold
      # @param top_k [Integer] Top-k sampling
      # @param do_sample [Boolean] Whether to sample or use greedy decoding
      # @return [Torch::Tensor] Generated token IDs
      def generate(input_ids, max_new_tokens: 100, temperature: 1.0, top_p: 0.9,
                   top_k: 50, do_sample: true, eos_token_id: nil, pad_token_id: nil)
        eval
        generated = input_ids.clone

        Torch.no_grad do
          max_new_tokens.times do
            # Forward pass
            outputs = forward(generated)
            next_token_logits = outputs[:logits][0.., -1, 0..]

            # Apply temperature
            next_token_logits = next_token_logits / temperature if temperature != 1.0

            if do_sample
              # Top-k filtering
              if top_k > 0
                indices_to_remove = next_token_logits < Torch.topk(next_token_logits, top_k).values[0.., -1, nil]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -Float::INFINITY)
              end

              # Top-p (nucleus) filtering
              if top_p < 1.0
                sorted_logits, sorted_indices = Torch.sort(next_token_logits, descending: true)
                cumulative_probs = Torch.cumsum(Torch::NN::Functional.softmax(sorted_logits, dim: -1), dim: -1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[0.., 1..] = sorted_indices_to_remove[0.., 0...-1].clone
                sorted_indices_to_remove[0.., 0] = false

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -Float::INFINITY)
              end

              # Sample
              probs = Torch::NN::Functional.softmax(next_token_logits, dim: -1)
              next_token = Torch.multinomial(probs, num_samples: 1)
            else
              # Greedy
              next_token = next_token_logits.argmax(dim: -1, keepdim: true)
            end

            # Append to generated
            generated = Torch.cat([generated, next_token], dim: 1)

            # Check for EOS
            if eos_token_id
              # Handle both single and array EOS token IDs
              eos_ids = eos_token_id.is_a?(Array) ? eos_token_id : [eos_token_id]
              next_token_val = next_token[0, 0].item rescue next_token.to(:int64)[0, 0].item
              break if eos_ids.include?(next_token_val)
            end
          end
        end

        generated
      end

      def save(path)
        FileUtils.mkdir_p(path)

        weights_path = File.join(path, "model.safetensors")
        Safetensors::Torch.save_file(state_dict, weights_path)

        save_config = @config.to_h.merge("model_type" => "causal_lm")

        config_path = File.join(path, "config.json")
        File.write(config_path, JSON.pretty_generate(save_config))
      end

      private

      def self.load_pretrained_weights(model, weights_path)
        # Load and copy weights one at a time to minimize memory usage
        model_state = model.state_dict
        model_keys = model_state.keys

        Torch.no_grad do
          Safetensors::Torch.load_file(weights_path).each do |name, tensor|
            mapped_name = map_llama_weight_name(name)
            if model_keys.include?(mapped_name)
              target = model_state[mapped_name]
              # Convert dtype if needed
              tensor = tensor.to(target.dtype) if tensor.dtype != target.dtype
              target.copy!(tensor)
            end
          end
        end

        # Force garbage collection to free loaded tensors
        GC.start
      end

      def self.load_sharded_weights(model, model_path)
        # Find all safetensors shards
        shards = Dir.glob(File.join(model_path, "model-*.safetensors")).sort

        return if shards.empty?

        model_state = model.state_dict
        model_keys = model_state.keys

        # Load each shard and copy weights immediately to minimize memory
        Torch.no_grad do
          shards.each do |shard_path|
            Safetensors::Torch.load_file(shard_path).each do |name, tensor|
              mapped_name = map_llama_weight_name(name)
              if model_keys.include?(mapped_name)
                target = model_state[mapped_name]
                tensor = tensor.to(target.dtype) if tensor.dtype != target.dtype
                target.copy!(tensor)
              end
            end
            # GC after each shard to free memory
            GC.start
          end
        end
      end

      def self.map_llama_weight_name(hf_name)
        name = hf_name.dup

        # Embeddings
        name = name.sub("model.embed_tokens", "decoder.embed_tokens")
        name = name.sub("lm_head", "lm_head")

        # Layers
        name = name.gsub("model.layers", "decoder.layers")

        # Attention (works for both Llama and Gemma)
        name = name.gsub(".self_attn.q_proj", ".self_attn.q_proj")
        name = name.gsub(".self_attn.k_proj", ".self_attn.k_proj")
        name = name.gsub(".self_attn.v_proj", ".self_attn.v_proj")
        name = name.gsub(".self_attn.o_proj", ".self_attn.o_proj")

        # Gemma 3 QK normalization
        name = name.gsub(".self_attn.q_norm", ".self_attn.q_norm")
        name = name.gsub(".self_attn.k_norm", ".self_attn.k_norm")

        # MLP
        name = name.gsub(".mlp.gate_proj", ".mlp.gate_proj")
        name = name.gsub(".mlp.up_proj", ".mlp.up_proj")
        name = name.gsub(".mlp.down_proj", ".mlp.down_proj")

        # Norms (standard)
        name = name.gsub(".input_layernorm", ".input_layernorm")
        name = name.gsub(".post_attention_layernorm", ".post_attention_layernorm")

        # Gemma 3 additional norms
        name = name.gsub(".pre_feedforward_layernorm", ".pre_feedforward_layernorm")
        name = name.gsub(".post_feedforward_layernorm", ".post_feedforward_layernorm")

        name = name.sub("model.norm", "decoder.norm")

        name
      end
    end
  end
end
