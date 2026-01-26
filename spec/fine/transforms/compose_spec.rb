# frozen_string_literal: true

RSpec.describe Fine::Transforms::Compose do
  describe "#call" do
    it "applies transforms in order" do
      results = []

      t1 = ->(x) { results << 1; x + 1 }
      t2 = ->(x) { results << 2; x * 2 }

      compose = described_class.new([t1, t2])
      result = compose.call(5)

      expect(results).to eq([1, 2])
      expect(result).to eq(12) # (5 + 1) * 2
    end
  end

  describe "#<<" do
    it "appends a transform" do
      compose = described_class.new([])
      compose << ->(x) { x + 1 }

      expect(compose.transforms.size).to eq(1)
    end
  end
end
