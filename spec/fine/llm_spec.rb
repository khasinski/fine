# frozen_string_literal: true

RSpec.describe Fine::LLM do
  describe "#initialize" do
    it "accepts a model ID" do
      llm = described_class.new("meta-llama/Llama-3.2-1B")
      expect(llm.model_id).to eq("meta-llama/Llama-3.2-1B")
    end

    it "accepts a configuration block" do
      llm = described_class.new("meta-llama/Llama-3.2-1B") do |config|
        config.epochs = 5
        config.batch_size = 2
        config.max_length = 1024
      end

      expect(llm.config.epochs).to eq(5)
      expect(llm.config.batch_size).to eq(2)
      expect(llm.config.max_length).to eq(1024)
    end
  end

  describe "#generate" do
    it "raises error when model not loaded" do
      llm = described_class.new("meta-llama/Llama-3.2-1B")

      expect { llm.generate("test") }
        .to raise_error(Fine::TrainingError, /not loaded/)
    end
  end

  describe "#save" do
    it "raises error when model not trained" do
      llm = described_class.new("meta-llama/Llama-3.2-1B")

      expect { llm.save("/tmp/model") }
        .to raise_error(Fine::TrainingError, /not trained/)
    end
  end

  describe Fine::LLMConfiguration do
    subject(:config) { described_class.new }

    it "has default max_length" do
      expect(config.max_length).to eq(2048)
    end

    it "has default batch_size" do
      expect(config.batch_size).to eq(4)
    end

    it "has default gradient_accumulation_steps" do
      expect(config.gradient_accumulation_steps).to eq(4)
    end

    it "has default max_grad_norm" do
      expect(config.max_grad_norm).to eq(1.0)
    end

    it "has default warmup_steps" do
      expect(config.warmup_steps).to eq(100)
    end

    it "has default freeze_layers" do
      expect(config.freeze_layers).to eq(0)
    end
  end
end
