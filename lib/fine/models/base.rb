# frozen_string_literal: true

module Fine
  module Models
    # Base class for all Fine models
    class Base < Torch::NN::Module
      attr_reader :config

      def initialize(config)
        super()
        @config = config
      end

      # Save model weights to a file
      #
      # @param path [String] Directory path to save to
      def save_pretrained(path)
        FileUtils.mkdir_p(path)

        # Save weights as safetensors
        weights_path = File.join(path, "model.safetensors")
        Safetensors::Torch.save_file(state_dict, weights_path)

        # Save config
        config_path = File.join(path, "config.json")
        File.write(config_path, JSON.pretty_generate(@config.to_h))
      end

      # Freeze all parameters (for feature extraction)
      def freeze!
        parameters.each { |p| p.requires_grad = false }
        self
      end

      # Unfreeze all parameters
      def unfreeze!
        parameters.each { |p| p.requires_grad = true }
        self
      end

      # Get number of trainable parameters
      def num_parameters(trainable_only: false)
        params = trainable_only ? parameters.select(&:requires_grad) : parameters
        params.sum { |p| p.numel }
      end
    end
  end
end
