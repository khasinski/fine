# frozen_string_literal: true

RSpec.describe Fine::Configuration do
  subject(:config) { described_class.new }

  describe "defaults" do
    it "sets default epochs" do
      expect(config.epochs).to eq(3)
    end

    it "sets default batch_size" do
      expect(config.batch_size).to eq(16)
    end

    it "sets default learning_rate" do
      expect(config.learning_rate).to eq(2e-5)
    end

    it "sets default optimizer" do
      expect(config.optimizer).to eq(:adamw)
    end

    it "sets default image_size" do
      expect(config.image_size).to eq(224)
    end
  end

  describe "#augmentation" do
    it "yields augmentation config" do
      config.augmentation do |aug|
        aug.random_horizontal_flip = true
      end

      expect(config.augmentation_config.random_horizontal_flip).to be true
    end
  end

  describe "#on_epoch_end" do
    it "registers a callback" do
      called = false
      config.on_epoch_end { called = true }

      expect(config.callbacks).not_to be_empty
      expect(config.callbacks.first).to be_a(Fine::Callbacks::LambdaCallback)
    end
  end
end
