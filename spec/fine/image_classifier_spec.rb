# frozen_string_literal: true

RSpec.describe Fine::ImageClassifier do
  describe "#initialize" do
    it "accepts a model ID" do
      classifier = described_class.new("google/siglip2-base-patch16-224")
      expect(classifier.model_id).to eq("google/siglip2-base-patch16-224")
    end

    it "accepts a configuration block" do
      classifier = described_class.new("google/siglip2-base-patch16-224") do |config|
        config.epochs = 10
        config.learning_rate = 1e-4
      end

      expect(classifier.config.epochs).to eq(10)
      expect(classifier.config.learning_rate).to eq(1e-4)
    end

    it "sets default image size" do
      classifier = described_class.new("google/siglip2-base-patch16-224")
      expect(classifier.config.image_size).to eq(224)
    end
  end

  describe "#predict" do
    it "raises error when model not trained" do
      classifier = described_class.new("google/siglip2-base-patch16-224")

      expect { classifier.predict("test.jpg") }
        .to raise_error(Fine::TrainingError, /not trained/)
    end
  end

  describe "#save" do
    it "raises error when model not trained" do
      classifier = described_class.new("google/siglip2-base-patch16-224")

      expect { classifier.save("/tmp/model") }
        .to raise_error(Fine::TrainingError, /not trained/)
    end
  end

  describe "#class_names" do
    it "returns empty array when no label map" do
      classifier = described_class.new("google/siglip2-base-patch16-224")
      expect(classifier.class_names).to eq([])
    end
  end
end
