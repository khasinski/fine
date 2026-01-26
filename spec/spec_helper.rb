# frozen_string_literal: true

require "bundler/setup"
require "fine"

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # Filter integration tests unless explicitly run
  config.filter_run_excluding :integration unless ENV["RUN_INTEGRATION"]

  # Filter network tests unless explicitly run
  config.filter_run_excluding :network unless ENV["RUN_NETWORK_TESTS"]
end

# Test fixtures path
FIXTURES_PATH = File.expand_path("fixtures", __dir__)
