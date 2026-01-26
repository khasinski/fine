# frozen_string_literal: true

RSpec.describe Fine::Callbacks::Base do
  describe "callback methods" do
    it "has on_train_begin" do
      callback = described_class.new
      expect(callback).to respond_to(:on_train_begin)
    end

    it "has on_train_end" do
      callback = described_class.new
      expect(callback).to respond_to(:on_train_end)
    end

    it "has on_epoch_begin" do
      callback = described_class.new
      expect(callback).to respond_to(:on_epoch_begin)
    end

    it "has on_epoch_end" do
      callback = described_class.new
      expect(callback).to respond_to(:on_epoch_end)
    end

    it "has on_batch_end" do
      callback = described_class.new
      expect(callback).to respond_to(:on_batch_end)
    end
  end
end

RSpec.describe Fine::Callbacks::LambdaCallback do
  it "calls the provided block on epoch end" do
    called_with = nil
    callback = described_class.new(on_epoch_end: ->(epoch, metrics) { called_with = [epoch, metrics] })

    callback.on_epoch_end(nil, 1, { loss: 0.5 })

    expect(called_with).to eq([1, { loss: 0.5 }])
  end
end
