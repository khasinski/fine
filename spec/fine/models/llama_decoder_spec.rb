# frozen_string_literal: true

RSpec.describe Fine::Models::RMSNorm do
  describe "#forward" do
    it "normalizes hidden states" do
      norm = described_class.new(64, eps: 1e-6)
      input = Torch.randn([2, 10, 64])

      output = norm.call(input)

      expect(output.shape).to eq([2, 10, 64])
    end

    it "preserves batch and sequence dimensions" do
      norm = described_class.new(128)
      input = Torch.randn([4, 20, 128])

      output = norm.call(input)

      expect(output.shape).to eq(input.shape)
    end
  end
end

RSpec.describe Fine::Models::LlamaMLP do
  describe "#forward" do
    it "transforms hidden states" do
      mlp = described_class.new(hidden_size: 64, intermediate_size: 128)
      input = Torch.randn([2, 10, 64])

      output = mlp.call(input)

      expect(output.shape).to eq([2, 10, 64])
    end
  end
end

RSpec.describe Fine::Models::RotaryEmbedding do
  describe "#call" do
    it "returns cos and sin embeddings" do
      rope = described_class.new(32, 512, 10000.0)
      x = Torch.randn([2, 4, 10, 32])  # batch, heads, seq, head_dim
      position_ids = Torch.arange(10).unsqueeze(0).expand(2, -1)

      cos, sin = rope.call(x, position_ids)

      expect(cos).to be_a(Torch::Tensor)
      expect(sin).to be_a(Torch::Tensor)
    end
  end
end
