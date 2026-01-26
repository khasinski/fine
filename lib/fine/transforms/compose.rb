# frozen_string_literal: true

module Fine
  module Transforms
    # Composes multiple transforms into a single callable
    class Compose
      attr_reader :transforms

      def initialize(transforms)
        @transforms = transforms
      end

      def call(image)
        @transforms.reduce(image) { |img, transform| transform.call(img) }
      end

      def <<(transform)
        @transforms << transform
        self
      end

      def prepend(transform)
        @transforms.unshift(transform)
        self
      end
    end
  end
end
