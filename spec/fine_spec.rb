# frozen_string_literal: true

RSpec.describe Fine do
  it "has a version number" do
    expect(Fine::VERSION).not_to be_nil
  end

  describe ".configure" do
    it "yields a configuration object" do
      Fine.configure do |config|
        expect(config).to be_a(Fine::GlobalConfiguration)
      end
    end

    it "sets configuration values" do
      Fine.configure do |config|
        config.cache_dir = "/tmp/fine-test"
        config.log_level = :debug
      end

      expect(Fine.configuration.cache_dir).to eq("/tmp/fine-test")
      expect(Fine.configuration.log_level).to eq(:debug)
    end
  end

  describe ".cache_dir" do
    it "returns the configured cache directory" do
      Fine.configure { |c| c.cache_dir = "/custom/path" }
      expect(Fine.cache_dir).to eq("/custom/path")
    end
  end
end
