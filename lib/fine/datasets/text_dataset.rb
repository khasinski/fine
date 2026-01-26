# frozen_string_literal: true

module Fine
  module Datasets
    # Dataset for text classification tasks
    #
    # Supports JSONL and CSV formats:
    #   JSONL: {"text": "...", "label": "positive"}
    #   CSV: text,label (with header)
    #
    class TextDataset
      include Enumerable

      attr_reader :texts, :labels, :label_map, :inverse_label_map, :tokenizer

      # Load dataset from a JSONL file
      #
      # @param path [String] Path to JSONL file
      # @param tokenizer [AutoTokenizer] Tokenizer to use
      # @param text_column [String] Name of text field
      # @param label_column [String] Name of label field
      # @return [TextDataset]
      def self.from_jsonl(path, tokenizer:, text_column: "text", label_column: "label")
        raise DatasetError, "File not found: #{path}" unless File.exist?(path)

        texts = []
        labels = []

        File.foreach(path) do |line|
          next if line.strip.empty?

          data = JSON.parse(line)
          texts << data[text_column]
          labels << data[label_column]
        end

        raise DatasetError, "No data found in #{path}" if texts.empty?

        new(texts: texts, labels: labels, tokenizer: tokenizer)
      end

      # Load dataset from a CSV file
      #
      # @param path [String] Path to CSV file
      # @param tokenizer [AutoTokenizer] Tokenizer to use
      # @param text_column [String] Name of text column
      # @param label_column [String] Name of label column
      # @return [TextDataset]
      def self.from_csv(path, tokenizer:, text_column: "text", label_column: "label")
        require "csv"
        raise DatasetError, "File not found: #{path}" unless File.exist?(path)

        texts = []
        labels = []

        CSV.foreach(path, headers: true) do |row|
          texts << row[text_column]
          labels << row[label_column]
        end

        raise DatasetError, "No data found in #{path}" if texts.empty?

        new(texts: texts, labels: labels, tokenizer: tokenizer)
      end

      # Load from file (auto-detect format)
      #
      # @param path [String] Path to data file
      # @param tokenizer [AutoTokenizer] Tokenizer to use
      # @return [TextDataset]
      def self.from_file(path, tokenizer:, **kwargs)
        case File.extname(path).downcase
        when ".jsonl", ".json"
          from_jsonl(path, tokenizer: tokenizer, **kwargs)
        when ".csv"
          from_csv(path, tokenizer: tokenizer, **kwargs)
        else
          # Try JSONL first, then CSV
          begin
            from_jsonl(path, tokenizer: tokenizer, **kwargs)
          rescue JSON::ParserError
            from_csv(path, tokenizer: tokenizer, **kwargs)
          end
        end
      end

      def initialize(texts:, labels:, tokenizer:, label_map: nil)
        raise ArgumentError, "texts and labels must have same length" if texts.size != labels.size

        @texts = texts
        @tokenizer = tokenizer

        # Build label map if not provided
        if label_map
          @label_map = label_map
        else
          unique_labels = labels.uniq.sort
          @label_map = unique_labels.each_with_index.to_h
        end

        # Convert string labels to integers
        @labels = labels.map do |label|
          label.is_a?(Integer) ? label : @label_map[label]
        end

        # Build inverse mapping
        @inverse_label_map = @label_map.invert
      end

      # Get a single item from the dataset
      #
      # @param index [Integer] Index of the item
      # @return [Hash] Hash with tokenized inputs and label
      def [](index)
        text = @texts[index]
        encoding = @tokenizer.encode(text, return_tensors: false)

        {
          input_ids: encoding[:input_ids].first,
          attention_mask: encoding[:attention_mask].first,
          token_type_ids: encoding[:token_type_ids]&.first,
          label: @labels[index]
        }.compact
      end

      # Number of items in the dataset
      def size
        @texts.size
      end
      alias length size

      # Iterate over all items
      def each
        return enum_for(:each) unless block_given?

        size.times { |i| yield self[i] }
      end

      # Number of classes
      def num_classes
        @label_map.size
      end

      # Get class names in order
      def class_names
        @inverse_label_map.sort.map(&:last)
      end

      # Split dataset into train and validation sets
      #
      # @param test_size [Float] Fraction of data for validation (0.0-1.0)
      # @param shuffle [Boolean] Whether to shuffle before splitting
      # @param stratify [Boolean] Whether to maintain class distribution
      # @param seed [Integer, nil] Random seed
      # @return [Array<TextDataset, TextDataset>] Train and validation datasets
      def split(test_size: 0.2, shuffle: true, stratify: true, seed: nil)
        rng = seed ? Random.new(seed) : Random.new

        indices = (0...size).to_a
        indices = indices.shuffle(random: rng) if shuffle && !stratify

        if stratify
          train_indices, val_indices = stratified_split(indices, test_size, rng)
        else
          split_idx = (size * (1 - test_size)).round
          train_indices = indices[0...split_idx]
          val_indices = indices[split_idx..]
        end

        train_set = subset(train_indices)
        val_set = subset(val_indices)

        [train_set, val_set]
      end

      private

      def subset(indices)
        TextDataset.new(
          texts: indices.map { |i| @texts[i] },
          labels: indices.map { |i| @labels[i] },
          tokenizer: @tokenizer,
          label_map: @label_map
        )
      end

      def stratified_split(indices, test_size, rng)
        train_indices = []
        val_indices = []

        # Group indices by label
        by_label = indices.group_by { |i| @labels[i] }

        by_label.each_value do |label_indices|
          shuffled = label_indices.shuffle(random: rng)
          split_idx = (shuffled.size * (1 - test_size)).round

          train_indices.concat(shuffled[0...split_idx])
          val_indices.concat(shuffled[split_idx..])
        end

        [train_indices.shuffle(random: rng), val_indices.shuffle(random: rng)]
      end
    end

    # Dataset for text pair tasks (similarity, NLI, etc.)
    class TextPairDataset
      include Enumerable

      attr_reader :texts_a, :texts_b, :labels, :tokenizer

      # Load from JSONL with query/positive pairs
      #
      # @param path [String] Path to JSONL file
      # @param tokenizer [AutoTokenizer] Tokenizer to use
      # @return [TextPairDataset]
      def self.from_jsonl(path, tokenizer:, text_a_column: "query", text_b_column: "positive")
        raise DatasetError, "File not found: #{path}" unless File.exist?(path)

        texts_a = []
        texts_b = []
        labels = []

        File.foreach(path) do |line|
          next if line.strip.empty?

          data = JSON.parse(line)
          texts_a << data[text_a_column]
          texts_b << data[text_b_column]
          labels << (data["label"] || 1.0)  # Default to positive pair
        end

        new(texts_a: texts_a, texts_b: texts_b, labels: labels, tokenizer: tokenizer)
      end

      def initialize(texts_a:, texts_b:, labels:, tokenizer:)
        @texts_a = texts_a
        @texts_b = texts_b
        @labels = labels
        @tokenizer = tokenizer
      end

      def [](index)
        encoding = @tokenizer.encode_pair(@texts_a[index], @texts_b[index], return_tensors: false)

        {
          input_ids: encoding[:input_ids].first,
          attention_mask: encoding[:attention_mask].first,
          token_type_ids: encoding[:token_type_ids]&.first,
          label: @labels[index]
        }.compact
      end

      def size
        @texts_a.size
      end
      alias length size

      def each
        return enum_for(:each) unless block_given?

        size.times { |i| yield self[i] }
      end
    end
  end
end
