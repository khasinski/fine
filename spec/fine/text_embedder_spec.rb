# frozen_string_literal: true

RSpec.describe Fine::TextEmbedder do
  describe Fine::EmbeddingConfiguration do
    subject(:config) { described_class.new }

    it "has default max_length" do
      expect(config.max_length).to eq(256)
    end

    it "has default pooling_mode" do
      expect(config.pooling_mode).to eq(:mean)
    end

    it "has default loss" do
      expect(config.loss).to eq(:multiple_negatives_ranking)
    end

    it "has default learning_rate" do
      expect(config.learning_rate).to eq(2e-5)
    end

    it "has default batch_size" do
      expect(config.batch_size).to eq(32)
    end
  end
end
