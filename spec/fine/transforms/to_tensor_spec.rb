# frozen_string_literal: true

RSpec.describe Fine::Transforms::ToTensor do
  let(:image_path) { File.join(FIXTURES_PATH, "images/cat/cat_1.jpg") }
  let(:image) { Vips::Image.new_from_file(image_path) }

  describe "#call" do
    it "converts Vips image to Torch tensor" do
      transform = described_class.new
      result = transform.call(image)

      expect(result).to be_a(Torch::Tensor)
    end

    it "creates tensor with correct shape (C, H, W)" do
      transform = described_class.new
      result = transform.call(image)

      expect(result.shape[0]).to eq(3)  # Channels
      expect(result.shape[1]).to eq(image.height)
      expect(result.shape[2]).to eq(image.width)
    end

    it "normalizes values to 0-1 range" do
      transform = described_class.new
      result = transform.call(image)

      expect(result.min.item).to be >= 0.0
      expect(result.max.item).to be <= 1.0
    end
  end
end
