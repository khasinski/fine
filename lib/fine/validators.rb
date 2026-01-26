# frozen_string_literal: true

module Fine
  # Data validation with helpful error messages
  module Validators
    class ValidationError < Error
      attr_reader :line_number, :expected_format

      def initialize(message, line_number: nil, expected_format: nil)
        @line_number = line_number
        @expected_format = expected_format
        super(build_message(message))
      end

      private

      def build_message(message)
        parts = [message]
        parts << "Line #{@line_number}" if @line_number
        parts << "\nExpected format:\n#{@expected_format}" if @expected_format
        parts.join(" ")
      end
    end

    class << self
      # Validate text classification data
      #
      # @param path [String] Path to JSONL file
      # @raise [ValidationError] if validation fails
      def validate_text_classification!(path)
        validate_file_exists!(path)
        validate_jsonl!(path) do |data, line_num|
          unless data.key?("text") || data.key?(:text)
            raise ValidationError.new(
              "Missing 'text' field",
              line_number: line_num,
              expected_format: TEXT_CLASSIFICATION_FORMAT
            )
          end
          unless data.key?("label") || data.key?(:label)
            raise ValidationError.new(
              "Missing 'label' field",
              line_number: line_num,
              expected_format: TEXT_CLASSIFICATION_FORMAT
            )
          end
        end
      end

      # Validate text pairs for embedding training
      #
      # @param path [String] Path to JSONL file
      # @raise [ValidationError] if validation fails
      def validate_text_pairs!(path)
        validate_file_exists!(path)
        validate_jsonl!(path) do |data, line_num|
          has_text_a = data.key?("text_a") || data.key?(:text_a) ||
                       data.key?("anchor") || data.key?(:anchor) ||
                       data.key?("sentence1") || data.key?(:sentence1) ||
                       data.key?("query") || data.key?(:query)
          has_text_b = data.key?("text_b") || data.key?(:text_b) ||
                       data.key?("positive") || data.key?(:positive) ||
                       data.key?("sentence2") || data.key?(:sentence2)

          unless has_text_a && has_text_b
            raise ValidationError.new(
              "Missing text pair fields",
              line_number: line_num,
              expected_format: TEXT_PAIRS_FORMAT
            )
          end
        end
      end

      # Validate instruction data for LLM fine-tuning
      #
      # @param path [String] Path to JSONL file
      # @param format [Symbol] Expected format (:alpaca, :sharegpt, :simple, :auto)
      # @raise [ValidationError] if validation fails
      def validate_instructions!(path, format: :auto)
        validate_file_exists!(path)

        first_line = File.open(path, &:readline)
        first_data = JSON.parse(first_line, symbolize_names: true)
        detected_format = format == :auto ? detect_instruction_format(first_data) : format

        validate_jsonl!(path) do |data, line_num|
          case detected_format
          when :alpaca
            validate_alpaca_format!(data, line_num)
          when :sharegpt
            validate_sharegpt_format!(data, line_num)
          when :simple
            validate_simple_format!(data, line_num)
          end
        end

        detected_format
      end

      # Validate image directory structure
      #
      # @param path [String] Path to directory
      # @raise [ValidationError] if validation fails
      def validate_image_directory!(path)
        unless File.directory?(path)
          raise ValidationError.new(
            "Directory not found: #{path}",
            expected_format: IMAGE_DIRECTORY_FORMAT
          )
        end

        subdirs = Dir.entries(path).reject { |e| e.start_with?(".") }
        subdirs = subdirs.select { |e| File.directory?(File.join(path, e)) }

        if subdirs.empty?
          raise ValidationError.new(
            "No class subdirectories found in #{path}",
            expected_format: IMAGE_DIRECTORY_FORMAT
          )
        end

        # Check each subdirectory has images
        subdirs.each do |subdir|
          subdir_path = File.join(path, subdir)
          images = Dir.glob(File.join(subdir_path, "*.{jpg,jpeg,png,gif,webp}"))
          if images.empty?
            raise ValidationError.new(
              "No images found in class directory: #{subdir_path}",
              expected_format: IMAGE_DIRECTORY_FORMAT
            )
          end
        end

        subdirs
      end

      # Quick check if file looks valid (non-blocking, for warnings)
      #
      # @param path [String] Path to file
      # @param type [Symbol] Type of data (:text_classification, :text_pairs, :instructions)
      # @return [Hash] { valid: true/false, warnings: [...], line_count: N }
      def check(path, type:)
        result = { valid: true, warnings: [], line_count: 0 }

        begin
          case type
          when :text_classification
            validate_text_classification!(path)
          when :text_pairs
            validate_text_pairs!(path)
          when :instructions
            validate_instructions!(path)
          when :image_directory
            validate_image_directory!(path)
          end

          result[:line_count] = File.readlines(path).count if File.file?(path)
        rescue ValidationError => e
          result[:valid] = false
          result[:warnings] << e.message
        rescue StandardError => e
          result[:valid] = false
          result[:warnings] << "Unexpected error: #{e.message}"
        end

        result
      end

      private

      def validate_file_exists!(path)
        unless File.exist?(path)
          raise ValidationError.new("File not found: #{path}")
        end

        if File.empty?(path)
          raise ValidationError.new("File is empty: #{path}")
        end
      end

      def validate_jsonl!(path)
        File.foreach(path).with_index(1) do |line, line_num|
          next if line.strip.empty?

          begin
            data = JSON.parse(line, symbolize_names: true)
          rescue JSON::ParserError => e
            raise ValidationError.new(
              "Invalid JSON: #{e.message}",
              line_number: line_num
            )
          end

          yield(data, line_num) if block_given?
        end
      end

      def detect_instruction_format(data)
        if data.key?(:instruction)
          :alpaca
        elsif data.key?(:conversations)
          :sharegpt
        elsif data.key?(:prompt) || data.key?(:text)
          :simple
        else
          raise ValidationError.new(
            "Cannot detect instruction format",
            expected_format: INSTRUCTION_FORMATS
          )
        end
      end

      def validate_alpaca_format!(data, line_num)
        unless data.key?(:instruction)
          raise ValidationError.new(
            "Missing 'instruction' field for Alpaca format",
            line_number: line_num,
            expected_format: ALPACA_FORMAT
          )
        end
        unless data.key?(:output) || data.key?(:response)
          raise ValidationError.new(
            "Missing 'output' or 'response' field for Alpaca format",
            line_number: line_num,
            expected_format: ALPACA_FORMAT
          )
        end
      end

      def validate_sharegpt_format!(data, line_num)
        unless data.key?(:conversations)
          raise ValidationError.new(
            "Missing 'conversations' field for ShareGPT format",
            line_number: line_num,
            expected_format: SHAREGPT_FORMAT
          )
        end
        unless data[:conversations].is_a?(Array)
          raise ValidationError.new(
            "'conversations' must be an array",
            line_number: line_num,
            expected_format: SHAREGPT_FORMAT
          )
        end
      end

      def validate_simple_format!(data, line_num)
        unless data.key?(:prompt) || data.key?(:text)
          raise ValidationError.new(
            "Missing 'prompt' or 'text' field for simple format",
            line_number: line_num,
            expected_format: SIMPLE_FORMAT
          )
        end
      end

      # Format examples for error messages
      TEXT_CLASSIFICATION_FORMAT = <<~FORMAT
        {"text": "This product is great!", "label": "positive"}
        {"text": "Terrible experience", "label": "negative"}
      FORMAT

      TEXT_PAIRS_FORMAT = <<~FORMAT
        {"text_a": "How do I reset my password?", "text_b": "Click forgot password on login page"}

        Alternative field names: query/positive, anchor/positive, sentence1/sentence2
      FORMAT

      ALPACA_FORMAT = <<~FORMAT
        {"instruction": "Summarize this text", "input": "Long text here...", "output": "Summary here"}
        {"instruction": "Translate to French", "output": "Bonjour"}
      FORMAT

      SHAREGPT_FORMAT = <<~FORMAT
        {"conversations": [
          {"from": "human", "value": "Hello"},
          {"from": "assistant", "value": "Hi there!"}
        ]}
      FORMAT

      SIMPLE_FORMAT = <<~FORMAT
        {"prompt": "Question here", "completion": "Answer here"}
        {"text": "Full text for language modeling"}
      FORMAT

      INSTRUCTION_FORMATS = <<~FORMAT
        Alpaca: {"instruction": "...", "output": "..."}
        ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
        Simple: {"prompt": "...", "completion": "..."}
      FORMAT

      IMAGE_DIRECTORY_FORMAT = <<~FORMAT
        data/
          cats/
            cat1.jpg
            cat2.jpg
          dogs/
            dog1.jpg
            dog2.jpg
      FORMAT
    end
  end
end
