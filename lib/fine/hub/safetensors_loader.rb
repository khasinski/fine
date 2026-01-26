# frozen_string_literal: true

module Fine
  module Hub
    # Loads SafeTensors weights into torch.rb models
    class SafetensorsLoader
      # Load weights from safetensors file into a model
      #
      # @param model [Torch::NN::Module] The model to load weights into
      # @param path [String] Path to the safetensors file
      # @param strict [Boolean] If true, raise error on missing/unexpected keys
      # @param prefix [String] Prefix to add/remove from weight names
      # @param skip_mapping [Boolean] If true, skip weight name mapping (for loading saved Fine models)
      # @return [Hash] Hash with :missing_keys and :unexpected_keys arrays
      def self.load_into_model(model, path, strict: false, prefix: nil, skip_mapping: false)
        tensors = Safetensors::Torch.load_file(path)

        # Get model's state dict keys
        model_keys = model.state_dict.keys

        # Map and filter tensors
        mapped_tensors = {}
        unexpected_keys = []

        tensors.each do |name, tensor|
          mapped_name = skip_mapping ? name : map_weight_name(name, prefix: prefix)

          if model_keys.include?(mapped_name)
            mapped_tensors[mapped_name] = tensor
          else
            unexpected_keys << name
          end
        end

        # Find missing keys
        missing_keys = model_keys - mapped_tensors.keys

        # Raise error if strict mode and there are issues
        if strict && (missing_keys.any? || unexpected_keys.any?)
          raise WeightLoadingError.new(
            "Weight loading failed",
            missing_keys: missing_keys,
            unexpected_keys: unexpected_keys
          )
        end

        # Load the mapped tensors by manually copying data
        # (torch.rb doesn't support strict: false yet)
        # Use no_grad to avoid in-place operation errors on leaf tensors
        Torch.no_grad do
          state_dict = model.state_dict
          mapped_tensors.each do |name, tensor|
            if state_dict.key?(name)
              state_dict[name].copy!(tensor)
            end
          end
        end

        { missing_keys: missing_keys, unexpected_keys: unexpected_keys }
      end

      # Map HuggingFace weight names to torch.rb model structure
      #
      # @param hf_name [String] Original HuggingFace weight name
      # @param prefix [String] Optional prefix to strip
      # @return [String] Mapped weight name for torch.rb
      def self.map_weight_name(hf_name, prefix: nil)
        name = hf_name.dup

        # Strip prefix if provided
        name = name.sub(/^#{Regexp.escape(prefix)}\.?/, "") if prefix

        # SigLIP2 specific mappings
        # Note: prefix (e.g., "vision_model") is already stripped above
        name = name.sub("embeddings.patch_embedding", "patch_embed.proj")
        name = name.sub("embeddings.position_embedding.weight", "pos_embed")
        name = name.sub("encoder.layers", "blocks")
        name = name.sub("post_layernorm", "norm")
        name = name.sub("head", "head")

        # Transformer block mappings
        name = name.gsub(".self_attn.", ".attn.")
        name = name.gsub(".layer_norm1.", ".norm1.")
        name = name.gsub(".layer_norm2.", ".norm2.")
        # mlp.fc1, mlp.fc2, q_proj, k_proj, v_proj, out_proj names match our model

        name
      end

      # Load raw tensors from safetensors file
      #
      # @param path [String] Path to safetensors file
      # @return [Hash<String, Torch::Tensor>] Hash of tensor name to tensor
      def self.load_file(path)
        Safetensors::Torch.load_file(path)
      end

      # List tensor names in a safetensors file
      #
      # @param path [String] Path to safetensors file
      # @return [Array<String>] List of tensor names
      def self.tensor_names(path)
        Safetensors.safe_open(path, framework: "torch") do |f|
          f.keys
        end
      end
    end
  end
end
