# frozen_string_literal: true

require_relative "lib/fine/version"

Gem::Specification.new do |spec|
  spec.name          = "fine"
  spec.version       = Fine::VERSION
  spec.authors       = ["Chris Hasinski"]
  spec.email         = ["krzysztof.hasinski@gmail.com"]

  spec.summary       = "Fine-tune ML models with Ruby"
  spec.description   = "A Ruby-native interface for fine-tuning machine learning models, starting with image classification using SigLIP2"
  spec.homepage      = "https://github.com/khasinski/fine"
  spec.license       = "MIT"

  spec.required_ruby_version = ">= 3.1"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (f == __FILE__) || f.match(%r{\A(?:(?:bin|test|spec|features)/|\.(?:git|travis|circleci)|appveyor)})
    end
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # Core ML dependencies
  spec.add_dependency "torch-rb", ">= 0.17"
  spec.add_dependency "safetensors", ">= 0.1"
  spec.add_dependency "tokenizers", ">= 0.4"

  # Image processing
  spec.add_dependency "ruby-vips", ">= 2.1"

  # Utilities
  spec.add_dependency "tty-progressbar", ">= 0.18"
  spec.add_dependency "down", ">= 5.0"

  # Web server (for `fine server` command)
  spec.add_dependency "sinatra", "~> 4.0"
  spec.add_dependency "puma", "~> 6.0"
  spec.add_dependency "rackup", "~> 2.1"

  # Development dependencies
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rspec", "~> 3.12"
  spec.add_development_dependency "rubocop", "~> 1.50"
end
