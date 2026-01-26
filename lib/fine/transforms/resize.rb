# frozen_string_literal: true

module Fine
  module Transforms
    # Resize image to specified dimensions
    class Resize
      attr_reader :width, :height, :interpolation

      # @param width [Integer] Target width
      # @param height [Integer, nil] Target height (defaults to width for square)
      # @param interpolation [Symbol] Interpolation method (:bilinear, :nearest, :bicubic)
      def initialize(width, height = nil, interpolation: :bilinear)
        @width = width
        @height = height || width
        @interpolation = interpolation
      end

      def call(image)
        # Map interpolation to vips kernel names
        vips_kernel = case @interpolation
                      when :nearest then :nearest
                      when :bilinear then :linear
                      when :bicubic then :cubic
                      else :linear
                      end

        # Calculate scale factors
        h_scale = @width.to_f / image.width
        v_scale = @height.to_f / image.height

        image.resize(h_scale, vscale: v_scale, kernel: vips_kernel)
      end
    end
  end
end
