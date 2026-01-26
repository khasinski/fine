# frozen_string_literal: true

module Fine
  module Callbacks
    # Progress bar callback using TTY::ProgressBar
    class ProgressBar < Base
      def initialize(show_epoch: true, show_batch: true)
        super()
        @show_epoch = show_epoch
        @show_batch = show_batch
        @epoch_bar = nil
        @batch_bar = nil
      end

      def on_train_begin(trainer)
        return unless @show_epoch

        @epoch_bar = TTY::ProgressBar.new(
          "Training [:bar] :current/:total epochs",
          total: trainer.config.epochs,
          width: 30
        )
      end

      def on_epoch_begin(trainer, epoch)
        return unless @show_batch

        @batch_bar = TTY::ProgressBar.new(
          "  Epoch #{epoch + 1} [:bar] :current/:total batches :rate/s",
          total: trainer.train_loader.size,
          width: 25,
          hide_cursor: true
        )
      end

      def on_batch_end(_trainer, _batch_idx, _loss)
        @batch_bar&.advance
      end

      def on_epoch_end(_trainer, epoch, metrics)
        @batch_bar&.finish

        # Format metrics for display
        metrics_str = metrics.map { |k, v| "#{k}: #{format_value(v)}" }.join(", ")
        puts "  #{metrics_str}"

        @epoch_bar&.advance
      end

      def on_train_end(_trainer)
        @epoch_bar&.finish
        puts "Training complete!"
      end

      private

      def format_value(v)
        case v
        when Float then format("%.4f", v)
        when Torch::Tensor then format("%.4f", v.item)
        else v.to_s
        end
      end
    end
  end
end
