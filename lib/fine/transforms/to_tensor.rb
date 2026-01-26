# frozen_string_literal: true

module Fine
  module Transforms
    # Convert a Vips::Image to a Torch::Tensor
    class ToTensor
      # @param scale [Boolean] If true, scale pixel values to [0, 1]
      def initialize(scale: true)
        @scale = scale
      end

      def call(image)
        # Get image as array of bytes
        # Vips images are (H, W, C) format

        # Ensure image is in RGB format
        image = ensure_rgb(image)

        # Get raw pixel data as a flat array
        width = image.width
        height = image.height
        bands = image.bands

        # Convert to array of floats
        data = image.write_to_memory.unpack("C*")

        # Create tensor with shape (H, W, C)
        tensor = Torch.tensor(data, dtype: :float32).reshape([height, width, bands])

        # Scale to [0, 1] if requested
        tensor = tensor / 255.0 if @scale

        # Permute to (C, H, W) format expected by PyTorch
        tensor.permute([2, 0, 1])
      end

      private

      def ensure_rgb(image)
        case image.bands
        when 1
          # Grayscale to RGB
          image.bandjoin([image, image])
        when 4
          # RGBA to RGB (drop alpha)
          image.extract_band(0, n: 3)
        else
          image
        end
      end
    end
  end
end
