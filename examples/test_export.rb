#!/usr/bin/env ruby
# frozen_string_literal: true

# Test export module

require "bundler/setup"
require "fine"

puts "Testing Fine::Export module..."
puts "=" * 50

puts "\n1. Testing GGUF quantization options..."
options = Fine::Export.gguf_quantization_options
puts "   Available quantization types: #{options.keys.join(', ')}"
options.each do |type, desc|
  puts "   - #{type}: #{desc}"
end

puts "\n2. Testing GGUF exporter constants..."
puts "   GGUF Magic: 0x#{Fine::Export::GGUFExporter::GGUF_MAGIC.to_s(16).upcase}"
puts "   GGUF Version: #{Fine::Export::GGUFExporter::GGUF_VERSION}"
puts "   Available quantizations: #{Fine::Export::GGUFExporter::QUANTIZATION_TYPES.keys.join(', ')}"

puts "\n3. Testing ONNX exporter..."
puts "   Supported types: #{Fine::Export::ONNXExporter::SUPPORTED_TYPES.map(&:to_s).join(', ')}"

puts "\n" + "=" * 50
puts "Export module tests passed!"
