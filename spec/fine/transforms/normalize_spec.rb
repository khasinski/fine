# frozen_string_literal: true

RSpec.describe Fine::Transforms::Normalize do
  describe "#call" do
    it "normalizes tensor values" do
      # Create a simple tensor (batch, channels, height, width)
      tensor = Torch.ones([1, 3, 4, 4])

      transform = described_class.new
      result = transform.call(tensor)

      # After normalization, values should be different from 1.0
      expect(result.mean.item).not_to eq(1.0)
    end

    it "accepts custom mean and std" do
      tensor = Torch.ones([1, 3, 4, 4])

      transform = described_class.new(
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5]
      )
      result = transform.call(tensor)

      # (1.0 - 0.5) / 0.5 = 1.0
      expect(result.mean.item).to be_within(0.01).of(1.0)
    end
  end
end
