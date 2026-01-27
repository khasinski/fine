# frozen_string_literal: true

module Fine
  module Server
    module Callbacks
      # Callback that broadcasts training events via SSE
      class WebCallback < Fine::Callbacks::Base
        def initialize(session)
          super()
          @session = session
          @start_time = nil
          @batch_count = 0
        end

        def on_train_begin(trainer)
          @start_time = Time.now
          @session.broadcast_event(:train_begin, {
            epochs: trainer.config.epochs,
            started_at: @start_time.iso8601
          })
        end

        def on_epoch_begin(trainer, epoch)
          @batch_count = 0
          @session.broadcast_event(:epoch_begin, {
            epoch: epoch + 1,
            total_epochs: trainer.config.epochs
          })
        end

        def on_batch_end(trainer, batch_idx, loss)
          @batch_count += 1
          loss_value = loss.is_a?(Numeric) ? loss : loss.item

          @session.metrics[:current_batch] = batch_idx
          @session.metrics[:current_loss] = loss_value

          @session.broadcast_event(:batch_end, {
            batch: batch_idx,
            loss: loss_value.round(6),
            timestamp: Time.now.iso8601
          })
        end

        def on_epoch_end(trainer, epoch, metrics)
          formatted = metrics.transform_values do |v|
            v.is_a?(Float) ? v.round(6) : v
          end

          @session.history << { epoch: epoch + 1, **formatted }
          @session.metrics[:current_epoch] = epoch + 1
          @session.metrics.merge!(formatted)

          elapsed = Time.now - @start_time

          @session.broadcast_event(:epoch_end, {
            epoch: epoch + 1,
            total_epochs: trainer.config.epochs,
            metrics: formatted,
            elapsed: elapsed.round(1)
          })
        end

        def on_train_end(trainer)
          elapsed = Time.now - @start_time

          @session.broadcast_event(:train_complete, {
            history: @session.history,
            elapsed: elapsed.round(1)
          })
        end
      end
    end
  end
end
