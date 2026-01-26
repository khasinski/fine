# frozen_string_literal: true

RSpec.describe Fine::Datasets::ImageDataset do
  let(:images_path) { File.join(FIXTURES_PATH, "images") }
  let(:transforms) do
    Fine::Transforms::Compose.new([
      Fine::Transforms::Resize.new(32),
      Fine::Transforms::ToTensor.new,
      Fine::Transforms::Normalize.new
    ])
  end

  describe ".from_directory" do
    it "loads images from subdirectories as classes" do
      dataset = described_class.from_directory(images_path, transforms: transforms)

      expect(dataset.size).to eq(4)  # 2 cats + 2 dogs
      expect(dataset.num_classes).to eq(2)
    end

    it "creates a label map" do
      dataset = described_class.from_directory(images_path, transforms: transforms)

      expect(dataset.label_map).to include("cat" => 0, "dog" => 1)
    end

    it "returns class names" do
      dataset = described_class.from_directory(images_path, transforms: transforms)

      expect(dataset.class_names).to contain_exactly("cat", "dog")
    end
  end

  describe "#[]" do
    it "returns image tensor and label" do
      dataset = described_class.from_directory(images_path, transforms: transforms)
      item = dataset[0]

      expect(item).to be_a(Hash)
      expect(item[:pixel_values]).to be_a(Torch::Tensor)
      expect(item[:label]).to be_a(Integer)
      expect(item[:label]).to be >= 0
      expect(item[:label]).to be < 2
    end

    it "applies transforms" do
      dataset = described_class.from_directory(images_path, transforms: transforms)
      item = dataset[0]

      # Should be (C, H, W) with size 32
      expect(item[:pixel_values].shape).to eq([3, 32, 32])
    end
  end

  describe "#split" do
    it "splits dataset into train and test" do
      dataset = described_class.from_directory(images_path, transforms: transforms)
      train, test = dataset.split(test_size: 0.5)

      expect(train.size).to eq(2)
      expect(test.size).to eq(2)
    end

    it "preserves label map" do
      dataset = described_class.from_directory(images_path, transforms: transforms)
      train, test = dataset.split(test_size: 0.5)

      expect(train.label_map).to eq(dataset.label_map)
      expect(test.label_map).to eq(dataset.label_map)
    end
  end
end
