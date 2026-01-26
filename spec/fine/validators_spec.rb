# frozen_string_literal: true

RSpec.describe Fine::Validators do
  let(:fixtures_path) { File.join(FIXTURES_PATH, "text") }
  let(:images_path) { File.join(FIXTURES_PATH, "images") }

  describe Fine::Validators::ValidationError do
    it "includes line number in message" do
      error = described_class.new("Test error", line_number: 5)
      expect(error.message).to include("Line 5")
    end

    it "includes expected format in message" do
      error = described_class.new("Test error", expected_format: '{"text": "...", "label": "..."}')
      expect(error.message).to include('{"text": "...", "label": "..."}')
    end

    it "exposes line_number and expected_format" do
      error = described_class.new("Test", line_number: 3, expected_format: "format")
      expect(error.line_number).to eq(3)
      expect(error.expected_format).to eq("format")
    end
  end

  describe ".validate_text_classification!" do
    it "accepts valid text classification data" do
      expect {
        described_class.validate_text_classification!(File.join(fixtures_path, "reviews.jsonl"))
      }.not_to raise_error
    end

    it "raises error for missing file" do
      expect {
        described_class.validate_text_classification!("/nonexistent/file.jsonl")
      }.to raise_error(Fine::Validators::ValidationError, /File not found/)
    end

    it "raises error for missing text field" do
      expect {
        described_class.validate_text_classification!(File.join(fixtures_path, "missing_text_field.jsonl"))
      }.to raise_error(Fine::Validators::ValidationError, /Missing 'text' field/)
    end

    it "raises error for missing label field" do
      expect {
        described_class.validate_text_classification!(File.join(fixtures_path, "missing_label_field.jsonl"))
      }.to raise_error(Fine::Validators::ValidationError, /Missing 'label' field/)
    end

    it "raises error for malformed JSON" do
      expect {
        described_class.validate_text_classification!(File.join(fixtures_path, "malformed.jsonl"))
      }.to raise_error(Fine::Validators::ValidationError, /Invalid JSON/)
    end

    it "raises error for empty file" do
      expect {
        described_class.validate_text_classification!(File.join(fixtures_path, "empty.jsonl"))
      }.to raise_error(Fine::Validators::ValidationError, /File is empty/)
    end

    it "includes line number for errors" do
      expect {
        described_class.validate_text_classification!(File.join(fixtures_path, "malformed.jsonl"))
      }.to raise_error(Fine::Validators::ValidationError, /Line 2/)
    end
  end

  describe ".validate_text_pairs!" do
    it "accepts valid text pairs data" do
      expect {
        described_class.validate_text_pairs!(File.join(fixtures_path, "pairs.jsonl"))
      }.not_to raise_error
    end

    it "raises error for missing file" do
      expect {
        described_class.validate_text_pairs!("/nonexistent/file.jsonl")
      }.to raise_error(Fine::Validators::ValidationError, /File not found/)
    end

    it "raises error for missing pair fields" do
      # reviews.jsonl has text/label, not text_a/text_b
      expect {
        described_class.validate_text_pairs!(File.join(fixtures_path, "reviews.jsonl"))
      }.to raise_error(Fine::Validators::ValidationError, /Missing text pair fields/)
    end
  end

  describe ".validate_instructions!" do
    context "with alpaca format" do
      it "accepts valid alpaca data" do
        result = described_class.validate_instructions!(
          File.join(fixtures_path, "instructions.jsonl"),
          format: :alpaca
        )
        expect(result).to eq(:alpaca)
      end

      it "raises error for missing instruction field" do
        expect {
          described_class.validate_instructions!(
            File.join(fixtures_path, "missing_instruction_field.jsonl"),
            format: :alpaca
          )
        }.to raise_error(Fine::Validators::ValidationError, /Missing 'instruction' field/)
      end
    end

    context "with sharegpt format" do
      it "accepts valid sharegpt data" do
        result = described_class.validate_instructions!(
          File.join(fixtures_path, "sharegpt.jsonl"),
          format: :sharegpt
        )
        expect(result).to eq(:sharegpt)
      end

      it "raises error for missing conversations field" do
        expect {
          described_class.validate_instructions!(
            File.join(fixtures_path, "reviews.jsonl"),
            format: :sharegpt
          )
        }.to raise_error(Fine::Validators::ValidationError, /Missing 'conversations' field/)
      end
    end

    context "with simple format" do
      it "accepts valid simple data" do
        result = described_class.validate_instructions!(
          File.join(fixtures_path, "simple_format.jsonl"),
          format: :simple
        )
        expect(result).to eq(:simple)
      end
    end

    context "with auto format detection" do
      it "detects alpaca format" do
        result = described_class.validate_instructions!(
          File.join(fixtures_path, "instructions.jsonl"),
          format: :auto
        )
        expect(result).to eq(:alpaca)
      end

      it "detects sharegpt format" do
        result = described_class.validate_instructions!(
          File.join(fixtures_path, "sharegpt.jsonl"),
          format: :auto
        )
        expect(result).to eq(:sharegpt)
      end

      it "detects simple format" do
        result = described_class.validate_instructions!(
          File.join(fixtures_path, "simple_format.jsonl"),
          format: :auto
        )
        expect(result).to eq(:simple)
      end
    end
  end

  describe ".validate_image_directory!" do
    it "accepts valid image directory" do
      result = described_class.validate_image_directory!(images_path)
      expect(result).to include("cat", "dog")
    end

    it "raises error for nonexistent directory" do
      expect {
        described_class.validate_image_directory!("/nonexistent/dir")
      }.to raise_error(Fine::Validators::ValidationError, /Directory not found/)
    end

    it "raises error for directory without subdirectories" do
      Dir.mktmpdir do |tmpdir|
        expect {
          described_class.validate_image_directory!(tmpdir)
        }.to raise_error(Fine::Validators::ValidationError, /No class subdirectories found/)
      end
    end

    it "raises error for empty class directories" do
      Dir.mktmpdir do |tmpdir|
        FileUtils.mkdir(File.join(tmpdir, "empty_class"))
        expect {
          described_class.validate_image_directory!(tmpdir)
        }.to raise_error(Fine::Validators::ValidationError, /No images found/)
      end
    end
  end

  describe ".check" do
    it "returns valid result for valid data" do
      result = described_class.check(
        File.join(fixtures_path, "reviews.jsonl"),
        type: :text_classification
      )

      expect(result[:valid]).to be true
      expect(result[:warnings]).to be_empty
      expect(result[:line_count]).to eq(8)
    end

    it "returns invalid result with warnings for invalid data" do
      result = described_class.check(
        File.join(fixtures_path, "malformed.jsonl"),
        type: :text_classification
      )

      expect(result[:valid]).to be false
      expect(result[:warnings]).not_to be_empty
    end

    it "handles image directories" do
      result = described_class.check(images_path, type: :image_directory)
      expect(result[:valid]).to be true
    end
  end
end
