# frozen_string_literal: true

module Fine
  module Datasets
    # DataLoader for batching dataset samples
    class DataLoader
      include Enumerable

      attr_reader :dataset, :batch_size, :shuffle, :drop_last

      # @param dataset [ImageDataset] The dataset to load from
      # @param batch_size [Integer] Number of samples per batch
      # @param shuffle [Boolean] Whether to shuffle indices each epoch
      # @param drop_last [Boolean] Whether to drop the last incomplete batch
      def initialize(dataset, batch_size:, shuffle: false, drop_last: false)
        @dataset = dataset
        @batch_size = batch_size
        @shuffle = shuffle
        @drop_last = drop_last
        @indices = nil
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

        # Stack pixel_values into a single tensor
        pixel_values = Torch.stack(samples.map { |s| s[:pixel_values] })

        # Stack labels into a single tensor
        labels = Torch.tensor(samples.map { |s| s[:label] }, dtype: :long)

        { pixel_values: pixel_values, labels: labels }
      end
    end
  end
end
