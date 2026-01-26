# frozen_string_literal: true

require "tokenizers"

module Fine
  module Tokenizers
    # Wrapper around HuggingFace tokenizers
    class AutoTokenizer
      attr_reader :tokenizer, :model_id, :max_length

      # Load tokenizer from a pretrained model
      #
      # @param model_id [String] HuggingFace model ID
      # @param max_length [Integer] Maximum sequence length
      # @return [AutoTokenizer]
      def self.from_pretrained(model_id, max_length: 512)
        new(model_id, max_length: max_length)
      end

      def initialize(model_id, max_length: 512)
        @model_id = model_id
        @max_length = max_length
        @tokenizer = load_tokenizer(model_id)

        configure_tokenizer
      end

      # Tokenize a single text or batch of texts
      #
      # @param texts [String, Array<String>] Text(s) to tokenize
      # @param padding [Boolean] Whether to pad sequences
      # @param truncation [Boolean] Whether to truncate sequences
      # @param return_tensors [Boolean] Whether to return Torch tensors
      # @return [Hash] Hash with :input_ids, :attention_mask, and optionally :token_type_ids
      def encode(texts, padding: true, truncation: true, return_tensors: true)
        texts = [texts] if texts.is_a?(String)
        single_input = texts.size == 1

        # Encode all texts
        encodings = texts.map do |text|
          @tokenizer.encode(text)
        end

        # Get max length in batch for padding
        max_len = if padding
          [encodings.map { |e| e.ids.length }.max, @max_length].min
        else
          @max_length
        end

        # Build output arrays
        input_ids = []
        attention_mask = []
        token_type_ids = []

        encodings.each do |encoding|
          ids = encoding.ids
          mask = encoding.attention_mask
          type_ids = encoding.type_ids rescue Array.new(ids.length, 0)

          # Truncate if needed
          if truncation && ids.length > max_len
            ids = ids[0...max_len]
            mask = mask[0...max_len]
            type_ids = type_ids[0...max_len]
          end

          # Pad if needed
          if padding && ids.length < max_len
            pad_length = max_len - ids.length
            ids = ids + Array.new(pad_length, pad_token_id)
            mask = mask + Array.new(pad_length, 0)
            type_ids = type_ids + Array.new(pad_length, 0)
          end

          input_ids << ids
          attention_mask << mask
          token_type_ids << type_ids
        end

        result = {
          input_ids: input_ids,
          attention_mask: attention_mask
        }

        # Only include token_type_ids for BERT-style models
        result[:token_type_ids] = token_type_ids if has_token_type_ids?

        if return_tensors
          result.transform_values! { |v| Torch.tensor(v, dtype: :long) }
        end

        result
      end

      # Encode a pair of texts (for sentence pair tasks)
      #
      # @param text_a [String] First text
      # @param text_b [String] Second text
      # @return [Hash] Tokenized output
      def encode_pair(text_a, text_b, **kwargs)
        encoding = @tokenizer.encode(text_a, text_b)

        ids = encoding.ids
        mask = encoding.attention_mask
        type_ids = encoding.type_ids rescue Array.new(ids.length, 0)

        # Truncate if needed
        if kwargs.fetch(:truncation, true) && ids.length > @max_length
          ids = ids[0...@max_length]
          mask = mask[0...@max_length]
          type_ids = type_ids[0...@max_length]
        end

        result = {
          input_ids: [ids],
          attention_mask: [mask]
        }
        result[:token_type_ids] = [type_ids] if has_token_type_ids?

        if kwargs.fetch(:return_tensors, true)
          result.transform_values! { |v| Torch.tensor(v, dtype: :long) }
        end

        result
      end

      # Decode token IDs back to text
      #
      # @param token_ids [Array<Integer>] Token IDs
      # @param skip_special_tokens [Boolean] Whether to skip special tokens
      # @return [String] Decoded text
      def decode(token_ids, skip_special_tokens: true)
        token_ids = token_ids.to_a if token_ids.respond_to?(:to_a)
        @tokenizer.decode(token_ids, skip_special_tokens: skip_special_tokens)
      end

      # Encode without padding (for generation)
      # Returns only the actual tokens, no padding
      #
      # @param text [String] Text to tokenize
      # @return [Array<Integer>] Token IDs
      def encode_for_generation(text)
        # Temporarily disable padding
        @tokenizer.no_padding
        encoding = @tokenizer.encode(text)
        ids = encoding.ids
        # Re-enable padding
        @tokenizer.enable_padding(length: @max_length)
        ids
      end

      # Get vocabulary size
      def vocab_size
        @tokenizer.vocab_size
      end

      # Get pad token ID
      def pad_token_id
        @tokenizer.token_to_id(@tokenizer.padding&.dig("pad_token") || "[PAD]") || 0
      end

      # Get CLS token ID
      def cls_token_id
        @tokenizer.token_to_id("[CLS]") || @tokenizer.token_to_id("<s>") || 0
      end

      # Get SEP token ID
      def sep_token_id
        @tokenizer.token_to_id("[SEP]") || @tokenizer.token_to_id("</s>") || 0
      end

      # Get EOS token ID
      def eos_token_id
        @tokenizer.token_to_id("</s>") || @tokenizer.token_to_id("[SEP]") || @tokenizer.token_to_id("<|endoftext|>") || 0
      end

      # Get BOS token ID
      def bos_token_id
        @tokenizer.token_to_id("<s>") || @tokenizer.token_to_id("[CLS]") || @tokenizer.token_to_id("<|startoftext|>") || 0
      end

      # Save tokenizer to directory
      def save(path)
        FileUtils.mkdir_p(path)
        @tokenizer.save(File.join(path, "tokenizer.json"))
      end

      private

      def load_tokenizer(model_id)
        # Check if it's a local path with tokenizer.json
        local_tokenizer_path = File.join(model_id, "tokenizer.json")
        if File.exist?(local_tokenizer_path)
          return ::Tokenizers::Tokenizer.from_file(local_tokenizer_path)
        end

        # Check if model_id itself is a tokenizer.json path
        if File.exist?(model_id) && model_id.end_with?("tokenizer.json")
          return ::Tokenizers::Tokenizer.from_file(model_id)
        end

        # Try to load from HuggingFace Hub
        ::Tokenizers::Tokenizer.from_pretrained(model_id)
      rescue StandardError => e
        raise ConfigurationError, "Failed to load tokenizer for #{model_id}: #{e.message}"
      end

      def configure_tokenizer
        # Enable truncation
        @tokenizer.enable_truncation(@max_length)

        # Enable padding
        @tokenizer.enable_padding(length: @max_length)
      end

      def has_token_type_ids?
        # BERT-style models use token_type_ids, RoBERTa/DistilBERT don't always need them
        @model_id.include?("bert") && !@model_id.include?("roberta")
      end
    end
  end
end
