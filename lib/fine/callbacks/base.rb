# frozen_string_literal: true

module Fine
  module Callbacks
    # Base class for training callbacks
    class Base
      def on_train_begin(trainer); end
      def on_train_end(trainer); end
      def on_epoch_begin(trainer, epoch); end
      def on_epoch_end(trainer, epoch, metrics); end
      def on_batch_begin(trainer, batch_idx); end
      def on_batch_end(trainer, batch_idx, loss); end
    end

    # Callback that wraps lambda functions
    class LambdaCallback < Base
      def initialize(on_train_begin: nil, on_train_end: nil,
                     on_epoch_begin: nil, on_epoch_end: nil,
                     on_batch_begin: nil, on_batch_end: nil)
        super()
        @on_train_begin_fn = on_train_begin
        @on_train_end_fn = on_train_end
        @on_epoch_begin_fn = on_epoch_begin
        @on_epoch_end_fn = on_epoch_end
        @on_batch_begin_fn = on_batch_begin
        @on_batch_end_fn = on_batch_end
      end

      def on_train_begin(trainer)
        @on_train_begin_fn&.call(trainer)
      end

      def on_train_end(trainer)
        @on_train_end_fn&.call(trainer)
      end

      def on_epoch_begin(trainer, epoch)
        @on_epoch_begin_fn&.call(epoch)
      end

      def on_epoch_end(trainer, epoch, metrics)
        @on_epoch_end_fn&.call(epoch, metrics)
      end

      def on_batch_begin(trainer, batch_idx)
        @on_batch_begin_fn&.call(batch_idx)
      end

      def on_batch_end(trainer, batch_idx, loss)
        @on_batch_end_fn&.call(batch_idx, loss)
      end
    end

    # Early stopping callback
    class EarlyStopping < Base
      attr_reader :patience, :monitor, :best_value, :wait

      def initialize(patience: 3, monitor: :val_loss, mode: :min)
        super()
        @patience = patience
        @monitor = monitor
        @mode = mode
        @best_value = nil
        @wait = 0
      end

      def on_epoch_end(trainer, _epoch, metrics)
        current = metrics[@monitor]
        return unless current

        if @best_value.nil? || improved?(current)
          @best_value = current
          @wait = 0
        else
          @wait += 1
          if @wait >= @patience
            puts "Early stopping triggered after #{@patience} epochs without improvement"
            trainer.stop_training = true
          end
        end
      end

      private

      def improved?(current)
        if @mode == :min
          current < @best_value
        else
          current > @best_value
        end
      end
    end

    # Model checkpoint callback
    class ModelCheckpoint < Base
      def initialize(path:, save_best_only: true, monitor: :val_loss, mode: :min)
        super()
        @path = path
        @save_best_only = save_best_only
        @monitor = monitor
        @mode = mode
        @best_value = nil
      end

      def on_epoch_end(trainer, epoch, metrics)
        current = metrics[@monitor]

        if @save_best_only
          return unless current

          if @best_value.nil? || improved?(current)
            @best_value = current
            save_checkpoint(trainer, epoch, metrics)
          end
        else
          save_checkpoint(trainer, epoch, metrics)
        end
      end

      private

      def improved?(current)
        if @mode == :min
          current < @best_value
        else
          current > @best_value
        end
      end

      def save_checkpoint(trainer, epoch, metrics)
        checkpoint_path = @path.include?("{epoch}") ?
          @path.gsub("{epoch}", epoch.to_s) :
          @path

        trainer.model.save(checkpoint_path, label_map: trainer.label_map)
        puts "Saved checkpoint to #{checkpoint_path}"
      end
    end
  end
end
