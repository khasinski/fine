# frozen_string_literal: true

module Fine
  # Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
  #
  # LoRA freezes the pretrained model weights and injects trainable
  # rank decomposition matrices into each layer, dramatically reducing
  # the number of trainable parameters.
  #
  # @example
  #   model = Fine::Models::CausalLM.from_pretrained("google/gemma-3-4b-it")
  #   lora_model = Fine::LoRA.apply(model, rank: 8, alpha: 16, target_modules: ["q_proj", "v_proj"])
  #   # Only LoRA parameters are trainable now
  #
  module LoRA
    # LoRA Linear layer that wraps an existing Linear layer
    class LoRALinear < Torch::NN::Module
      attr_reader :in_features, :out_features, :rank, :alpha, :scaling

      def initialize(original_layer, rank: 8, alpha: 16, dropout: 0.0)
        super()

        @in_features = original_layer.weight.shape[1]
        @out_features = original_layer.weight.shape[0]
        @rank = rank
        @alpha = alpha
        @scaling = alpha.to_f / rank

        # Match dtype of original layer
        @dtype = original_layer.weight.dtype
        @device = original_layer.weight.device

        # Store original layer (frozen)
        @original = original_layer
        @original.weight.requires_grad = false
        @original.bias&.requires_grad = false if @original.respond_to?(:bias) && @original.bias

        # LoRA matrices A and B - match dtype of original layer
        # W' = W + (B @ A) * scaling
        # A: (rank, in_features) - initialized with Kaiming uniform
        # B: (out_features, rank) - initialized with zeros
        @lora_a = Torch::NN::Parameter.new(
          Torch.empty(@rank, @in_features, dtype: @dtype, device: @device)
        )
        @lora_b = Torch::NN::Parameter.new(
          Torch.zeros(@out_features, @rank, dtype: @dtype, device: @device)
        )

        # Initialize A with Kaiming uniform (in float32, then convert)
        temp_a = Torch.empty(@rank, @in_features)
        Torch::NN::Init.kaiming_uniform!(temp_a, a: Math.sqrt(5))
        @lora_a.data.copy!(temp_a.to(@dtype))

        # Optional dropout
        @dropout = dropout > 0 ? Torch::NN::Dropout.new(p: dropout) : nil
      end

      def forward(x)
        # Original forward pass (frozen)
        original_out = @original.call(x)

        # LoRA forward: x @ A.T @ B.T * scaling
        lora_out = x
        lora_out = @dropout.call(lora_out) if @dropout
        lora_out = lora_out.matmul(@lora_a.t)
        lora_out = lora_out.matmul(@lora_b.t)
        lora_out = lora_out * @scaling

        original_out + lora_out
      end

      # Number of trainable parameters
      def trainable_params
        @rank * @in_features + @out_features * @rank
      end

      # Merge LoRA weights into original layer (for inference)
      def merge!
        Torch.no_grad do
          delta_w = @lora_b.matmul(@lora_a) * @scaling
          @original.weight.add!(delta_w)
        end
      end
    end

    class << self
      # Apply LoRA to a model
      #
      # @param model [Torch::NN::Module] Model to apply LoRA to
      # @param rank [Integer] LoRA rank (lower = fewer params, higher = more capacity)
      # @param alpha [Integer] LoRA alpha (scaling factor)
      # @param dropout [Float] Dropout probability for LoRA layers
      # @param target_modules [Array<String>] Module names to apply LoRA to
      # @return [Torch::NN::Module] Model with LoRA applied
      def apply(model, rank: 8, alpha: 16, dropout: 0.0, target_modules: nil)
        raise ArgumentError, "Model cannot be nil for LoRA.apply" if model.nil?

        target_modules ||= default_target_modules

        # First freeze all parameters
        model.parameters.each { |p| p.requires_grad = false }

        # Track replacements
        replacements = []
        total_lora_params = 0

        # Find and replace target modules
        find_modules(model, target_modules) do |parent, name, layer|
          next unless layer.is_a?(Torch::NN::Linear)

          lora_layer = LoRALinear.new(layer, rank: rank, alpha: alpha, dropout: dropout)
          replacements << [parent, name, lora_layer]
          total_lora_params += lora_layer.trainable_params
        end

        # Apply replacements
        replacements.each do |parent, name, lora_layer|
          parent.instance_variable_set("@#{name}", lora_layer)
        end

        # Calculate stats
        total_params = count_params(model)
        trainable = count_trainable_params(model)

        puts "   LoRA applied to #{replacements.size} layers"
        puts "   Total params: #{format_params(total_params)}"
        puts "   Trainable params: #{format_params(trainable)} (#{(trainable.to_f / total_params * 100).round(2)}%)"

        model
      end

      # Merge LoRA weights into base model (for efficient inference)
      def merge!(model)
        find_lora_layers(model) do |lora_layer|
          lora_layer.merge!
        end
        model
      end

      # Get only trainable (LoRA) parameters
      def trainable_parameters(model)
        params = []
        find_lora_layers(model) do |lora_layer|
          params << lora_layer.lora_a
          params << lora_layer.lora_b
        end
        params
      end

      # Default modules to apply LoRA to (attention projections)
      def default_target_modules
        %w[q_proj k_proj v_proj o_proj]
      end

      private

      def find_modules(model, target_names, parent = nil, prefix = "", &block)
        model.instance_variables.each do |ivar|
          name = ivar.to_s.delete_prefix("@")
          child = model.instance_variable_get(ivar)

          if child.is_a?(Torch::NN::Module)
            full_name = prefix.empty? ? name : "#{prefix}.#{name}"

            if target_names.any? { |t| name == t || name.end_with?(t) }
              yield(model, name, child)
            end

            # Recurse into ModuleList
            if child.is_a?(Torch::NN::ModuleList)
              child.each_with_index do |layer, idx|
                find_modules(layer, target_names, child, "#{full_name}[#{idx}]", &block)
              end
            else
              find_modules(child, target_names, model, full_name, &block)
            end
          end
        end
      end

      def find_lora_layers(model, &block)
        model.instance_variables.each do |ivar|
          child = model.instance_variable_get(ivar)

          if child.is_a?(LoRALinear)
            yield(child)
          elsif child.is_a?(Torch::NN::ModuleList)
            child.each { |layer| find_lora_layers(layer, &block) }
          elsif child.is_a?(Torch::NN::Module)
            find_lora_layers(child, &block)
          end
        end
      end

      def count_params(model)
        model.parameters.sum { |p| p.numel }
      end

      def count_trainable_params(model)
        model.parameters.select { |p| p.requires_grad }.sum { |p| p.numel }
      end

      def format_params(n)
        if n >= 1_000_000_000
          "#{(n / 1_000_000_000.0).round(2)}B"
        elsif n >= 1_000_000
          "#{(n / 1_000_000.0).round(2)}M"
        elsif n >= 1_000
          "#{(n / 1_000.0).round(2)}K"
        else
          n.to_s
        end
      end
    end
  end
end
