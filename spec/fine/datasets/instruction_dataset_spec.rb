# frozen_string_literal: true

RSpec.describe Fine::Datasets::InstructionDataset do
  let(:instructions_path) { File.join(FIXTURES_PATH, "text/instructions.jsonl") }

  # Mock tokenizer for testing
  let(:mock_tokenizer) do
    tokenizer = double("Tokenizer")
    allow(tokenizer).to receive(:encode) do |text|
      tokens = text.split.take(20)
      {
        input_ids: [tokens.map { |t| t.hash.abs % 1000 }]
      }
    end
    tokenizer
  end

  describe ".from_jsonl" do
    it "loads instructions from JSONL file" do
      dataset = described_class.from_jsonl(
        instructions_path,
        tokenizer: mock_tokenizer,
        format: :alpaca
      )

      expect(dataset.size).to eq(4)
    end
  end

  describe "#[]" do
    it "returns input_ids, labels, and attention_mask" do
      dataset = described_class.from_jsonl(
        instructions_path,
        tokenizer: mock_tokenizer,
        format: :alpaca
      )

      item = dataset[0]

      expect(item).to have_key(:input_ids)
      expect(item).to have_key(:labels)
      expect(item).to have_key(:attention_mask)
    end

    it "returns tensors" do
      dataset = described_class.from_jsonl(
        instructions_path,
        tokenizer: mock_tokenizer,
        format: :alpaca
      )

      item = dataset[0]

      expect(item[:input_ids]).to be_a(Torch::Tensor)
      expect(item[:labels]).to be_a(Torch::Tensor)
    end
  end

  describe "#split" do
    it "splits dataset into train and test" do
      dataset = described_class.from_jsonl(
        instructions_path,
        tokenizer: mock_tokenizer,
        format: :alpaca
      )

      train, test = dataset.split(test_size: 0.5)

      expect(train.size).to eq(2)
      expect(test.size).to eq(2)
    end
  end

  describe "format detection" do
    it "detects alpaca format" do
      dataset = described_class.from_jsonl(
        instructions_path,
        tokenizer: mock_tokenizer,
        format: :auto
      )

      expect(dataset.size).to eq(4)
    end
  end
end
