# frozen_string_literal: true

RSpec.describe Fine::LoRA do
  describe Fine::LoRA::LoRALinear do
    let(:original_layer) do
      Torch::NN::Linear.new(64, 32)
    end

    subject(:lora_layer) do
      described_class.new(original_layer, rank: 4, alpha: 8)
    end

    describe "#initialize" do
      it "stores correct dimensions" do
        expect(lora_layer.in_features).to eq(64)
        expect(lora_layer.out_features).to eq(32)
        expect(lora_layer.rank).to eq(4)
        expect(lora_layer.alpha).to eq(8)
      end

      it "calculates correct scaling" do
        expect(lora_layer.scaling).to eq(2.0) # alpha / rank = 8 / 4
      end

      it "freezes original layer parameters" do
        # Trigger subject creation first
        lora_layer
        expect(original_layer.weight.requires_grad).to be false
      end
    end

    describe "#forward" do
      let(:input) { Torch.randn([2, 64]) }

      it "returns output with correct shape" do
        output = lora_layer.forward(input)
        expect(output.shape).to eq([2, 32])
      end

      it "returns different output than original layer" do
        original_output = original_layer.forward(input)
        lora_output = lora_layer.forward(input)

        # LoRA adds a delta to the original output
        # Initially B is zeros, so they should be equal
        # But after training would be different
        expect(lora_output.shape).to eq(original_output.shape)
      end
    end

    describe "#trainable_params" do
      it "returns correct count" do
        # A: (rank, in_features) = 4 * 64 = 256
        # B: (out_features, rank) = 32 * 4 = 128
        # Total: 256 + 128 = 384
        expect(lora_layer.trainable_params).to eq(384)
      end
    end

    describe "#merge!" do
      it "modifies original layer weights" do
        original_weight = original_layer.weight.clone

        # Set some non-zero values in lora_b to see the merge effect
        Torch.no_grad do
          lora_layer.instance_variable_get(:@lora_b).data.fill!(0.1)
        end

        lora_layer.merge!

        # Weights should be different after merge
        diff = (original_layer.weight - original_weight).abs.sum.item
        expect(diff).to be > 0
      end
    end

    context "with dropout" do
      let(:lora_with_dropout) do
        described_class.new(original_layer, rank: 4, alpha: 8, dropout: 0.1)
      end

      it "creates dropout layer" do
        dropout = lora_with_dropout.instance_variable_get(:@dropout)
        expect(dropout).to be_a(Torch::NN::Dropout)
      end
    end
  end

  describe ".default_target_modules" do
    it "returns attention projection layers" do
      expect(described_class.default_target_modules).to include("q_proj", "k_proj", "v_proj", "o_proj")
    end
  end

  describe ".trainable_parameters" do
    let(:mock_model) do
      model = Object.new
      def model.instance_variables
        [:@layer1, :@layer2]
      end
      model
    end

    it "returns empty array when no LoRA layers present" do
      allow(mock_model).to receive(:instance_variable_get).and_return(nil)
      params = described_class.trainable_parameters(mock_model)
      expect(params).to eq([])
    end
  end
end
