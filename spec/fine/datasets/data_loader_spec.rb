# frozen_string_literal: true

RSpec.describe Fine::Datasets::DataLoader do
  let(:images_path) { File.join(FIXTURES_PATH, "images") }
  let(:transforms) do
    Fine::Transforms::Compose.new([
      Fine::Transforms::Resize.new(32),
      Fine::Transforms::ToTensor.new,
      Fine::Transforms::Normalize.new
    ])
  end
  let(:dataset) { Fine::Datasets::ImageDataset.from_directory(images_path, transforms: transforms) }

  describe "#each" do
    it "yields batches" do
      loader = described_class.new(dataset, batch_size: 2)
      batches = loader.to_a

      expect(batches.size).to eq(2)  # 4 images / batch_size 2
    end

    it "returns tensors in batches" do
      loader = described_class.new(dataset, batch_size: 2)
      batch = loader.first

      expect(batch).to be_a(Hash)
      expect(batch[:pixel_values]).to be_a(Torch::Tensor)
      expect(batch[:labels]).to be_a(Torch::Tensor)
      expect(batch[:pixel_values].shape[0]).to eq(2)  # batch size
      expect(batch[:labels].shape[0]).to eq(2)
    end
  end

  describe "#size" do
    it "returns number of batches" do
      loader = described_class.new(dataset, batch_size: 2)
      expect(loader.size).to eq(2)
    end

    it "handles partial batches" do
      loader = described_class.new(dataset, batch_size: 3)
      expect(loader.size).to eq(2)  # ceil(4/3) = 2
    end
  end

  describe "shuffling" do
    it "can be disabled" do
      loader1 = described_class.new(dataset, batch_size: 4, shuffle: false)
      loader2 = described_class.new(dataset, batch_size: 4, shuffle: false)

      _, labels1 = loader1.first
      _, labels2 = loader2.first

      expect(labels1.to_a).to eq(labels2.to_a)
    end
  end
end
