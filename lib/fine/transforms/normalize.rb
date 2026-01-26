# frozen_string_literal: true

module Fine
  module Transforms
    # Normalize a tensor with mean and standard deviation
    class Normalize
      # ImageNet normalization values (commonly used)
      IMAGENET_MEAN = [0.485, 0.456, 0.406].freeze
      IMAGENET_STD = [0.229, 0.224, 0.225].freeze

      attr_reader :mean, :std

      # @param mean [Array<Float>] Mean values for each channel
      # @param std [Array<Float>] Standard deviation for each channel
      def initialize(mean: IMAGENET_MEAN, std: IMAGENET_STD)
        @mean = mean
        @std = std
      end

      def call(tensor)
        # Expect tensor shape: (C, H, W)
        raise ArgumentError, "Expected tensor, got #{tensor.class}" unless tensor.is_a?(Torch::Tensor)

        # Convert mean and std to tensors with shape (C, 1, 1)
        mean_tensor = Torch.tensor(@mean).view(-1, 1, 1)
        std_tensor = Torch.tensor(@std).view(-1, 1, 1)

        # Normalize: (x - mean) / std
        (tensor - mean_tensor) / std_tensor
      end
    end
  end
end
