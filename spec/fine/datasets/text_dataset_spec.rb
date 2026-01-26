# frozen_string_literal: true

RSpec.describe Fine::Datasets::TextDataset do
  let(:reviews_path) { File.join(FIXTURES_PATH, "text/reviews.jsonl") }

  # Mock tokenizer for testing
  let(:mock_tokenizer) do
    tokenizer = double("Tokenizer")
    allow(tokenizer).to receive(:encode) do |texts|
      texts = [texts] if texts.is_a?(String)
      {
        input_ids: texts.map { |t| (0...10).to_a },
        attention_mask: texts.map { |_| [1] * 10 },
        token_type_ids: texts.map { |_| [0] * 10 }
      }
    end
    tokenizer
  end

  describe ".from_file" do
    it "loads data from JSONL file" do
      dataset = described_class.from_file(reviews_path, tokenizer: mock_tokenizer)

      expect(dataset.size).to eq(8)
    end

    it "extracts labels" do
      dataset = described_class.from_file(reviews_path, tokenizer: mock_tokenizer)

      expect(dataset.num_classes).to eq(2)
      expect(dataset.label_map).to include("positive", "negative")
    end
  end

  describe "#[]" do
    it "returns tokenized data and label" do
      dataset = described_class.from_file(reviews_path, tokenizer: mock_tokenizer)
      item = dataset[0]

      expect(item).to have_key(:input_ids)
      expect(item).to have_key(:attention_mask)
      expect(item).to have_key(:label)
    end
  end

  describe "#split" do
    it "splits dataset" do
      dataset = described_class.from_file(reviews_path, tokenizer: mock_tokenizer)
      train, test = dataset.split(test_size: 0.25)

      expect(train.size).to eq(6)
      expect(test.size).to eq(2)
    end
  end
end
