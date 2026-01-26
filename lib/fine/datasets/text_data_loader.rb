# frozen_string_literal: true

module Fine
  module Datasets
    # DataLoader for text datasets with dynamic padding
    class TextDataLoader
      include Enumerable

      attr_reader :dataset, :batch_size, :shuffle, :drop_last

      def initialize(dataset, batch_size:, shuffle: false, drop_last: false)
        @dataset = dataset
        @batch_size = batch_size
        @shuffle = shuffle
        @drop_last = drop_last
      end

      # Iterate over batches
      def each_batch
        return enum_for(:each_batch) unless block_given?

        indices = (0...@dataset.size).to_a
        indices.shuffle! if @shuffle

        indices.each_slice(@batch_size) do |batch_indices|
          next if @drop_last && batch_indices.size < @batch_size

          yield collate(batch_indices)
        end
      end

      alias each each_batch

      # Number of batches
      def size
        n = @dataset.size / @batch_size
        n += 1 unless @drop_last || (@dataset.size % @batch_size).zero?
        n
      end

      alias num_batches size

      private

      def collate(indices)
        samples = indices.map { |i| @dataset[i] }

        # Find max length in this batch for dynamic padding
        max_len = samples.map { |s| s[:input_ids].length }.max

        # Pad and stack
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []

        samples.each do |sample|
          ids = sample[:input_ids]
          mask = sample[:attention_mask]
          type_ids = sample[:token_type_ids]

          # Pad to max_len
          pad_len = max_len - ids.length
          if pad_len > 0
            ids = ids + Array.new(pad_len, 0)
            mask = mask + Array.new(pad_len, 0)
            type_ids = type_ids + Array.new(pad_len, 0) if type_ids
          end

          input_ids << ids
          attention_mask << mask
          token_type_ids << type_ids if type_ids
          labels << sample[:label]
        end

        result = {
          input_ids: Torch.tensor(input_ids, dtype: :long),
          attention_mask: Torch.tensor(attention_mask, dtype: :long),
          labels: Torch.tensor(labels, dtype: :long)
        }

        result[:token_type_ids] = Torch.tensor(token_type_ids, dtype: :long) if token_type_ids.first

        result
      end
    end
  end
end
