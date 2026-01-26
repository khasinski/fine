# frozen_string_literal: true

module Fine
  module Training
    # Main training orchestrator
    class Trainer
      attr_reader :model, :config, :train_loader, :val_loader, :label_map
      attr_accessor :stop_training

      def initialize(model, config, train_dataset:, val_dataset: nil)
        @model = model
        @config = config
        @stop_training = false
        @label_map = train_dataset.label_map

        # Create data loaders
        @train_loader = Datasets::DataLoader.new(
          train_dataset,
          batch_size: config.batch_size,
          shuffle: true
        )

        @val_loader = if val_dataset
          Datasets::DataLoader.new(
            val_dataset,
            batch_size: config.batch_size,
            shuffle: false
          )
        end

        # History tracking
        @history = []
      end

      # Train the model
      #
      # @return [Array<Hash>] Training history (metrics per epoch)
      def fit
        @model.train

        # Build optimizer
        optimizer = build_optimizer
        scheduler = build_scheduler(optimizer)

        run_callbacks(:on_train_begin, self)

        @config.epochs.times do |epoch|
          break if @stop_training

          run_callbacks(:on_epoch_begin, self, epoch)

          # Training epoch
          train_metrics = train_epoch(optimizer, epoch)

          # Validation
          val_metrics = @val_loader ? evaluate : {}

          # Step scheduler
          scheduler&.step

          # Combine metrics
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
      #
      # @return [Hash] Evaluation metrics
      def evaluate
        @model.eval

        total_loss = 0.0
        correct = 0
        total = 0

        Torch.no_grad do
          @val_loader.each_batch do |batch|
            output = @model.call(batch[:pixel_values], labels: batch[:labels])

            total_loss += output[:loss].item * batch[:labels].size(0)
            predictions = output[:logits].argmax(dim: 1)
            correct += (predictions == batch[:labels]).sum.item
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

      def train_epoch(optimizer, epoch)
        total_loss = 0.0
        correct = 0
        total = 0

        @train_loader.each_batch.with_index do |batch, batch_idx|
          run_callbacks(:on_batch_begin, self, batch_idx)

          # Zero gradients
          optimizer.zero_grad

          # Forward pass
          output = @model.call(batch[:pixel_values], labels: batch[:labels])
          loss = output[:loss]

          # Backward pass
          loss.backward

          # Update weights
          optimizer.step

          # Track metrics
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
          Torch::Optim::SGD.new(
            params,
            lr: @config.learning_rate,
            momentum: 0.9
          )
        else
          raise ConfigurationError, "Unknown optimizer: #{@config.optimizer}"
        end
      end

      def build_scheduler(optimizer)
        return nil unless @config.scheduler

        case @config.scheduler
        when :cosine
          Torch::Optim::LRScheduler::CosineAnnealingLR.new(
            optimizer,
            @config.epochs
          )
        when :step
          Torch::Optim::LRScheduler::StepLR.new(
            optimizer,
            step_size: @config.epochs / 3,
            gamma: 0.1
          )
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
  end
end
