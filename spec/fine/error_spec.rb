# frozen_string_literal: true

RSpec.describe Fine::Error do
  it "inherits from StandardError" do
    expect(described_class.superclass).to eq(StandardError)
  end
end

RSpec.describe Fine::ModelNotFoundError do
  it "stores the model_id" do
    error = described_class.new("some/model")
    expect(error.model_id).to eq("some/model")
  end

  it "has a default message" do
    error = described_class.new("some/model")
    expect(error.message).to include("some/model")
  end

  it "accepts a custom message" do
    error = described_class.new("some/model", "Custom message")
    expect(error.message).to eq("Custom message")
  end
end

RSpec.describe Fine::ConfigurationError do
  it "inherits from Fine::Error" do
    expect(described_class.superclass).to eq(Fine::Error)
  end
end

RSpec.describe Fine::DatasetError do
  it "inherits from Fine::Error" do
    expect(described_class.superclass).to eq(Fine::Error)
  end
end

RSpec.describe Fine::TrainingError do
  it "inherits from Fine::Error" do
    expect(described_class.superclass).to eq(Fine::Error)
  end
end

RSpec.describe Fine::WeightLoadingError do
  it "stores missing and unexpected keys" do
    error = described_class.new(
      "Weight loading failed",
      missing_keys: ["layer1.weight"],
      unexpected_keys: ["layer2.bias"]
    )

    expect(error.missing_keys).to eq(["layer1.weight"])
    expect(error.unexpected_keys).to eq(["layer2.bias"])
  end
end

RSpec.describe Fine::ImageProcessingError do
  it "stores the path" do
    error = described_class.new("/path/to/image.jpg")
    expect(error.path).to eq("/path/to/image.jpg")
  end
end

RSpec.describe Fine::ExportError do
  it "inherits from Fine::Error" do
    expect(described_class.superclass).to eq(Fine::Error)
  end
end
