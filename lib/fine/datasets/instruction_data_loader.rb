# frozen_string_literal: true

module Fine
  module Datasets
    # DataLoader for instruction datasets with dynamic padding
    class InstructionDataLoader
      include Enumerable

      attr_reader :dataset, :batch_size, :shuffle, :pad_token_id

      def initialize(dataset, batch_size:, shuffle: false, pad_token_id: 0)
        @dataset = dataset
        @batch_size = batch_size
        @shuffle = shuffle
        @pad_token_id = pad_token_id
      end

      # Iterate over batches
      def each_batch
        return enum_for(:each_batch) unless block_given?

        indices = (0...@dataset.size).to_a
        indices.shuffle! if @shuffle

        indices.each_slice(@batch_size) do |batch_indices|
          yield collate(batch_indices)
        end
      end

      alias each each_batch

      # Number of batches
      def size
        (@dataset.size.to_f / @batch_size).ceil
      end

      alias num_batches size

      private

      def collate(indices)
        samples = indices.map { |i| @dataset[i] }

        # Find max length in this batch
        max_len = samples.map { |s| s[:input_ids].size(-1) }.max

        # Pad and stack
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        samples.each do |sample|
          ids = sample[:input_ids].squeeze(0).to_a
          mask = sample[:attention_mask].squeeze(0).to_a
          lbls = sample[:labels].squeeze(0).to_a

          # Pad to max_len (right padding)
          pad_len = max_len - ids.length
          if pad_len > 0
            ids = ids + Array.new(pad_len, @pad_token_id)
            mask = mask + Array.new(pad_len, 0)
            lbls = lbls + Array.new(pad_len, -100) # -100 is ignored in loss
          end

          input_ids_list << ids
          attention_mask_list << mask
          labels_list << lbls
        end

        {
          input_ids: Torch.tensor(input_ids_list, dtype: :long),
          attention_mask: Torch.tensor(attention_mask_list, dtype: :long),
          labels: Torch.tensor(labels_list, dtype: :long)
        }
      end
    end
  end
end
