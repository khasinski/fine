# frozen_string_literal: true

module Fine
  module Models
    # Simple classification head (linear layer)
    class ClassificationHead < Torch::NN::Module
      attr_reader :in_features, :num_classes

      def initialize(in_features, num_classes, dropout: 0.0)
        super()
        @in_features = in_features
        @num_classes = num_classes

        @dropout = Torch::NN::Dropout.new(p: dropout) if dropout.positive?
        @classifier = Torch::NN::Linear.new(in_features, num_classes)
      end

      def forward(x)
        x = @dropout.call(x) if @dropout
        @classifier.call(x)
      end
    end
  end
end
