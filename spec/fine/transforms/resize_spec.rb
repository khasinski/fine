# frozen_string_literal: true

RSpec.describe Fine::Transforms::Resize do
  let(:image_path) { File.join(FIXTURES_PATH, "images/cat/cat_1.jpg") }
  let(:image) { Vips::Image.new_from_file(image_path) }

  describe "#call" do
    it "resizes image to specified size" do
      transform = described_class.new(64)
      result = transform.call(image)

      expect(result.width).to eq(64)
      expect(result.height).to eq(64)
    end

    it "handles different sizes" do
      transform = described_class.new(128)
      result = transform.call(image)

      expect(result.width).to eq(128)
      expect(result.height).to eq(128)
    end
  end
end
