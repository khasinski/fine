# frozen_string_literal: true

RSpec.describe Fine::Export do
  describe ".gguf_quantization_options" do
    it "returns available quantization types" do
      options = described_class.gguf_quantization_options

      expect(options).to include(:f16, :q4_0, :q8_0)
      expect(options[:f16]).to be_a(String)
    end
  end
end

RSpec.describe Fine::Export::GGUFExporter do
  describe "GGUF constants" do
    it "defines GGUF magic number" do
      expect(described_class::GGUF_MAGIC).to eq(0x46554747)
    end

    it "defines quantization types" do
      expect(described_class::QUANTIZATION_TYPES).to include(:f16, :q4_0, :q8_0)
    end
  end
end

RSpec.describe Fine::Export::ONNXExporter do
  describe "supported types" do
    it "supports text classifier" do
      expect(described_class::SUPPORTED_TYPES).to include(Fine::TextClassifier)
    end

    it "supports text embedder" do
      expect(described_class::SUPPORTED_TYPES).to include(Fine::TextEmbedder)
    end

    it "supports image classifier" do
      expect(described_class::SUPPORTED_TYPES).to include(Fine::ImageClassifier)
    end

    it "supports LLM" do
      expect(described_class::SUPPORTED_TYPES).to include(Fine::LLM)
    end
  end
end
