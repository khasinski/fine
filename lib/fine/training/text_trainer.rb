# frozen_string_literal: true

module Fine
  module Training
    # Trainer for text classification models
    class TextTrainer
      attr_reader :model, :config, :train_loader, :val_loader, :label_map
      attr_accessor :stop_training

      def initialize(model, config, train_dataset:, val_dataset: nil)
        @model = model
        @config = config
        @stop_training = false
        @label_map = train_dataset.label_map

        @train_loader = Datasets::TextDataLoader.new(
          train_dataset,
          batch_size: config.batch_size,
          shuffle: true
        )

        @val_loader = if val_dataset
          Datasets::TextDataLoader.new(
            val_dataset,
            batch_size: config.batch_size,
            shuffle: false
          )
        end

        @history = []
      end

      # Train the model
      #
      # @return [Array<Hash>] Training history
      def fit
        @model.train

        optimizer = build_optimizer
        scheduler = build_scheduler(optimizer)

        run_callbacks(:on_train_begin, self)

        @config.epochs.times do |epoch|
          break if @stop_training

          run_callbacks(:on_epoch_begin, self, epoch)

          train_metrics = train_epoch(optimizer, epoch)

          val_metrics = @val_loader ? evaluate : {}

          scheduler&.step

          metrics = train_metrics.merge(
            val_metrics.transform_keys { |k| :"val_#{k}" }
          )
          @history << metrics

          run_callbacks(:on_epoch_end, self, epoch, metrics)
        end

        run_callbacks(:on_train_end, self)

        @history
      end

      # Evaluate on validation set
      def evaluate
        @model.eval

        total_loss = 0.0
        correct = 0
        total = 0

        Torch.no_grad do
          @val_loader.each_batch do |batch|
            output = @model.call(
              batch[:input_ids],
              attention_mask: batch[:attention_mask],
              token_type_ids: batch[:token_type_ids],
              labels: batch[:labels]
            )

            total_loss += output[:loss].item * batch[:labels].size(0)
            predictions = output[:logits].argmax(dim: 1)
            correct += predictions.eq(batch[:labels]).sum.item
            total += batch[:labels].size(0)
          end
        end

        @model.train

        {
          loss: total_loss / total,
          accuracy: correct.to_f / total
        }
      end

      private

      def train_epoch(optimizer, _epoch)
        total_loss = 0.0
        correct = 0
        total = 0

        @train_loader.each_batch.with_index do |batch, batch_idx|
          run_callbacks(:on_batch_begin, self, batch_idx)

          optimizer.zero_grad

          output = @model.call(
            batch[:input_ids],
            attention_mask: batch[:attention_mask],
            token_type_ids: batch[:token_type_ids],
            labels: batch[:labels]
          )

          loss = output[:loss]
          loss.backward
          optimizer.step

          batch_loss = loss.item
          total_loss += batch_loss * batch[:labels].size(0)

          predictions = output[:logits].argmax(dim: 1)
          correct += predictions.eq(batch[:labels]).sum.item
          total += batch[:labels].size(0)

          run_callbacks(:on_batch_end, self, batch_idx, batch_loss)
        end

        {
          loss: total_loss / total,
          accuracy: correct.to_f / total
        }
      end

      def build_optimizer
        params = @model.parameters.select(&:requires_grad)

        case @config.optimizer
        when :adam
          Torch::Optim::Adam.new(params, lr: @config.learning_rate)
        when :adamw
          Torch::Optim::AdamW.new(
            params,
            lr: @config.learning_rate,
            weight_decay: @config.weight_decay
          )
        when :sgd
          Torch::Optim::SGD.new(params, lr: @config.learning_rate, momentum: 0.9)
        else
          raise ConfigurationError, "Unknown optimizer: #{@config.optimizer}"
        end
      end

      def build_scheduler(optimizer)
        return nil unless @config.scheduler

        case @config.scheduler
        when :cosine
          Torch::Optim::LRScheduler::CosineAnnealingLR.new(optimizer, @config.epochs)
        when :linear
          # Linear decay
          Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.9)
        else
          nil
        end
      end

      def run_callbacks(method, *args)
        @config.callbacks.each do |callback|
          callback.send(method, *args)
        end
      end
    end

    # Trainer for text embedding models (contrastive learning)
    class EmbeddingTrainer
      attr_reader :model, :config, :train_loader
      attr_accessor :stop_training

      def initialize(model, config, train_dataset:)
        @model = model
        @config = config
        @stop_training = false

        @train_loader = Datasets::TextDataLoader.new(
          train_dataset,
          batch_size: config.batch_size,
          shuffle: true
        )

        @history = []
      end

      def fit
        @model.train

        optimizer = build_optimizer

        run_callbacks(:on_train_begin, self)

        @config.epochs.times do |epoch|
          break if @stop_training

          run_callbacks(:on_epoch_begin, self, epoch)

          metrics = train_epoch(optimizer)
          @history << metrics

          run_callbacks(:on_epoch_end, self, epoch, metrics)
        end

        run_callbacks(:on_train_end, self)

        @history
      end

      private

      def train_epoch(optimizer)
        total_loss = 0.0
        num_batches = 0

        @train_loader.each_batch do |batch|
          optimizer.zero_grad

          # For pair datasets, we get anchor and positive texts
          # Use forward() directly during training (not encode() which uses no_grad)
          output = @model.forward(
            batch[:input_ids],
            attention_mask: batch[:attention_mask]
          )
          embeddings = output[:embeddings]

          # Multiple Negatives Ranking Loss
          # Treat other samples in batch as negatives
          loss = multiple_negatives_ranking_loss(embeddings)

          loss.backward
          optimizer.step

          total_loss += loss.item
          num_batches += 1
        end

        { loss: total_loss / num_batches }
      end

      def multiple_negatives_ranking_loss(embeddings, scale: 20.0)
        # Split embeddings into anchors and positives (assuming paired data)
        batch_size = embeddings.size(0) / 2
        anchors = embeddings[0...batch_size]
        positives = embeddings[batch_size..]

        # Compute similarity matrix
        scores = Torch.matmul(anchors, positives.transpose(0, 1)) * scale

        # Labels: diagonal is positive (index i matches index i)
        labels = Torch.arange(batch_size, device: embeddings.device)

        Torch::NN::Functional.cross_entropy(scores, labels)
      end

      def build_optimizer
        params = @model.parameters.select(&:requires_grad)
        Torch::Optim::AdamW.new(params, lr: @config.learning_rate, weight_decay: @config.weight_decay)
      end

      def run_callbacks(method, *args)
        @config.callbacks.each { |cb| cb.send(method, *args) }
      end
    end
  end
end
