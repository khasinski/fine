# frozen_string_literal: true

RSpec.describe Fine::TextClassifier do
  describe "#initialize" do
    it "accepts a model ID" do
      classifier = described_class.new("distilbert-base-uncased")
      expect(classifier.model_id).to eq("distilbert-base-uncased")
    end

    it "accepts a configuration block" do
      classifier = described_class.new("distilbert-base-uncased") do |config|
        config.epochs = 5
        config.batch_size = 8
      end

      expect(classifier.config.epochs).to eq(5)
      expect(classifier.config.batch_size).to eq(8)
    end

    it "has default progress bar callback" do
      classifier = described_class.new("distilbert-base-uncased")
      expect(classifier.config.callbacks).not_to be_empty
    end
  end

  describe Fine::TextConfiguration do
    subject(:config) { described_class.new }

    it "has default max_length" do
      expect(config.max_length).to eq(256)
    end

    it "has default warmup_ratio" do
      expect(config.warmup_ratio).to eq(0.1)
    end

    it "has lower default learning rate than base" do
      expect(config.learning_rate).to eq(2e-5)
    end
  end
end
