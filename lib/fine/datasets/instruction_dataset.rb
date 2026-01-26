# frozen_string_literal: true

module Fine
  module Datasets
    # Dataset for instruction/chat fine-tuning
    #
    # Supports common formats:
    # - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    # - ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    # - Simple: {"prompt": "...", "completion": "..."}
    class InstructionDataset
      attr_reader :examples, :tokenizer, :max_length

      # Load from JSONL file
      #
      # @param path [String] Path to JSONL file
      # @param tokenizer [Tokenizers::AutoTokenizer] Tokenizer
      # @param format [Symbol] Data format (:alpaca, :sharegpt, :simple, :auto)
      # @param max_length [Integer] Maximum sequence length
      # @param validate [Boolean] Whether to validate the file first
      # @return [InstructionDataset]
      #
      # @example Alpaca format
      #   # {"instruction": "Summarize this", "input": "Long text...", "output": "Summary"}
      #   dataset = InstructionDataset.from_jsonl("data.jsonl", tokenizer: tok)
      #
      # @example ShareGPT format
      #   # {"conversations": [{"from": "human", "value": "Hi"}, {"from": "assistant", "value": "Hello!"}]}
      #   dataset = InstructionDataset.from_jsonl("chat.jsonl", tokenizer: tok, format: :sharegpt)
      #
      def self.from_jsonl(path, tokenizer:, format: :auto, max_length: 2048, validate: true)
        detected_format = Validators.validate_instructions!(path, format: format) if validate
        format = detected_format if validate && format == :auto

        examples = File.readlines(path).reject { |l| l.strip.empty? }.map do |line|
          JSON.parse(line, symbolize_names: true)
        end
        new(examples, tokenizer: tokenizer, format: format, max_length: max_length)
      end

      def initialize(examples, tokenizer:, format: :auto, max_length: 2048)
        @tokenizer = tokenizer
        @max_length = max_length
        @format = format == :auto ? detect_format(examples.first) : format

        @examples = examples.map { |ex| normalize_example(ex) }
      end

      def size
        @examples.size
      end

      def [](idx)
        example = @examples[idx]

        # Format as prompt + completion
        text = format_example(example)

        # Tokenize (without tensors for easier manipulation)
        encoding = @tokenizer.encode(text, return_tensors: false)
        input_ids = encoding[:input_ids].first

        # Get prompt length before truncation
        prompt_text = format_prompt_only(example)
        prompt_encoding = @tokenizer.encode(prompt_text, return_tensors: false)
        prompt_length = prompt_encoding[:input_ids].first.size

        # Truncate if needed, but ensure at least some completion tokens remain
        if input_ids.size > @max_length
          # If prompt alone is too long, truncate from the left (keep completion)
          if prompt_length >= @max_length - 10
            # Keep last max_length tokens (includes completion)
            input_ids = input_ids.last(@max_length)
            # No masking since we dropped the prompt prefix
            prompt_length = 0
          else
            input_ids = input_ids.first(@max_length)
          end
        end

        # Labels are same as input_ids for causal LM (predict next token)
        labels = input_ids.dup

        # Mask prompt tokens with -100 (ignored in loss), but only if there's room for completion
        if prompt_length > 0 && prompt_length < input_ids.size
          labels[0...prompt_length] = [-100] * prompt_length
        end
        # If prompt_length >= input_ids.size, don't mask (train on full sequence)

        {
          input_ids: Torch.tensor([input_ids]),
          labels: Torch.tensor([labels]),
          attention_mask: Torch.ones(1, input_ids.size)
        }
      end

      # Split dataset
      def split(test_size: 0.1, seed: 42)
        rng = Random.new(seed)
        indices = (0...size).to_a.shuffle(random: rng)

        split_idx = (size * (1 - test_size)).to_i
        train_indices = indices[0...split_idx]
        test_indices = indices[split_idx..]

        train_examples = train_indices.map { |i| @examples[i] }
        test_examples = test_indices.map { |i| @examples[i] }

        [
          self.class.new(train_examples, tokenizer: @tokenizer, format: @format, max_length: @max_length),
          self.class.new(test_examples, tokenizer: @tokenizer, format: @format, max_length: @max_length)
        ]
      end

      private

      def detect_format(example)
        if example.key?(:instruction)
          :alpaca
        elsif example.key?(:conversations)
          :sharegpt
        elsif example.key?(:prompt) || example.key?(:text)
          :simple
        else
          raise DatasetError, "Cannot detect format. Keys: #{example.keys}"
        end
      end

      def normalize_example(example)
        case @format
        when :alpaca
          {
            prompt: build_alpaca_prompt(example[:instruction], example[:input]),
            completion: example[:output] || example[:response]
          }
        when :sharegpt
          conversations = example[:conversations]
          # Take first human/assistant pair
          human = conversations.find { |c| c[:from] == "human" }
          assistant = conversations.find { |c| c[:from] == "gpt" || c[:from] == "assistant" }
          {
            prompt: human[:value],
            completion: assistant[:value]
          }
        when :simple
          {
            prompt: example[:prompt] || example[:text],
            completion: example[:completion] || example[:response] || ""
          }
        else
          raise DatasetError, "Unknown format: #{@format}"
        end
      end

      def build_alpaca_prompt(instruction, input = nil)
        if input && !input.empty?
          "### Instruction:\n#{instruction}\n\n### Input:\n#{input}\n\n### Response:\n"
        else
          "### Instruction:\n#{instruction}\n\n### Response:\n"
        end
      end

      def format_example(example)
        "#{example[:prompt]}#{example[:completion]}"
      end

      def format_prompt_only(example)
        example[:prompt]
      end
    end

    # Data loader for instruction dataset with dynamic padding
    class InstructionDataLoader
      include Enumerable

      def initialize(dataset, batch_size:, shuffle: true, pad_token_id: 0)
        @dataset = dataset
        @batch_size = batch_size
        @shuffle = shuffle
        @pad_token_id = pad_token_id
      end

      def each
        indices = (0...@dataset.size).to_a
        indices.shuffle! if @shuffle

        indices.each_slice(@batch_size) do |batch_indices|
          batch = batch_indices.map { |i| @dataset[i] }
          yield collate_batch(batch)
        end
      end

      def size
        (@dataset.size.to_f / @batch_size).ceil
      end

      private

      def collate_batch(batch)
        max_len = batch.map { |b| b[:input_ids].size(-1) }.max

        input_ids = []
        labels = []
        attention_masks = []

        batch.each do |item|
          seq_len = item[:input_ids].size(-1)
          pad_len = max_len - seq_len

          if pad_len > 0
            # Pad on the right
            input_ids << Torch.cat([
              item[:input_ids],
              Torch.full([1, pad_len], @pad_token_id)
            ], dim: 1)

            labels << Torch.cat([
              item[:labels],
              Torch.full([1, pad_len], -100)  # Ignore padding in loss
            ], dim: 1)

            attention_masks << Torch.cat([
              item[:attention_mask],
              Torch.zeros(1, pad_len)
            ], dim: 1)
          else
            input_ids << item[:input_ids]
            labels << item[:labels]
            attention_masks << item[:attention_mask]
          end
        end

        {
          input_ids: Torch.cat(input_ids, dim: 0),
          labels: Torch.cat(labels, dim: 0),
          attention_mask: Torch.cat(attention_masks, dim: 0)
        }
      end
    end
  end
end
