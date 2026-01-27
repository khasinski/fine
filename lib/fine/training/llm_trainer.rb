# frozen_string_literal: true

module Fine
  module Training
    # Trainer for causal language model fine-tuning
    class LLMTrainer
      attr_reader :model, :config, :train_dataset, :val_dataset, :train_loader, :history

      def initialize(model, config, train_dataset:, val_dataset: nil)
        @model = model
        @config = config
        @train_dataset = train_dataset
        @val_dataset = val_dataset
        @device = Fine.device
        @history = []
        @train_loader = nil
      end

      def fit
        @model.to(@device)
        @model.train

        # Create train loader early so callbacks can access it
        @train_loader = create_data_loader(@train_dataset, shuffle: true)

        optimizer = create_optimizer
        scheduler = create_scheduler(optimizer)

        @config.callbacks.each { |cb| cb.on_train_begin(self) }

        @config.epochs.times do |epoch|
          epoch_loss = train_epoch(optimizer, scheduler, epoch)

          metrics = { loss: epoch_loss }

          # Validation
          if @val_dataset
            val_loss, val_perplexity = evaluate
            metrics[:val_loss] = val_loss
            metrics[:val_perplexity] = val_perplexity
          end

          @history << { epoch: epoch + 1, **metrics }

          @config.callbacks.each { |cb| cb.on_epoch_end(self, epoch + 1, metrics) }
        end

        @config.callbacks.each { |cb| cb.on_train_end(self) }

        @history
      end

      def evaluate
        @model.eval
        total_loss = 0.0
        num_batches = 0

        data_loader = create_data_loader(@val_dataset, shuffle: false)

        Torch.no_grad do
          data_loader.each do |batch|
            batch = move_to_device(batch)

            outputs = @model.forward(
              batch[:input_ids],
              attention_mask: batch[:attention_mask],
              labels: batch[:labels]
            )

            total_loss += outputs[:loss].to(:float32).item
            num_batches += 1
          end
        end

        @model.train

        avg_loss = total_loss / num_batches
        perplexity = Math.exp(avg_loss)

        [avg_loss, perplexity]
      end

      private

      def train_epoch(optimizer, scheduler, epoch)
        total_loss = 0.0
        num_batches = 0

        total_steps = @train_loader.size

        @config.callbacks.each do |cb|
          cb.on_epoch_begin(self, epoch) if cb.respond_to?(:on_epoch_begin)
        end

        @train_loader.each_with_index do |batch, step|
          # Move batch to device
          input_ids = batch[:input_ids].to(@device)
          attention_mask = batch[:attention_mask].to(@device)
          labels = batch[:labels].to(@device)

          # Forward pass
          outputs = @model.forward(input_ids, attention_mask: attention_mask, labels: labels)

          # Get loss value before backward
          loss_value = outputs[:loss].detach.to(:float32).item

          # Backward pass - scale loss for gradient accumulation
          scaled_loss = outputs[:loss] / @config.gradient_accumulation_steps
          scaled_loss.backward

          # CRITICAL: Clear ALL references to free computation graph
          scaled_loss = nil
          outputs = nil
          input_ids = nil
          attention_mask = nil
          labels = nil
          batch = nil

          if (step + 1) % @config.gradient_accumulation_steps == 0
            # Gradient clipping
            if @config.max_grad_norm
              clip_grad_norm(@model.parameters, @config.max_grad_norm)
            end

            optimizer.step
            scheduler&.step
            optimizer.zero_grad

            # Force GC after each optimizer step to free computation graphs
            GC.start
          end

          total_loss += loss_value
          num_batches += 1

          @config.callbacks.each do |cb|
            cb.on_batch_end(self, step + 1, loss_value) if cb.respond_to?(:on_batch_end)
          end
        end

        total_loss / num_batches
      end

      def create_data_loader(dataset, shuffle:)
        Datasets::InstructionDataLoader.new(
          dataset,
          batch_size: @config.batch_size,
          shuffle: shuffle,
          pad_token_id: @config.pad_token_id || 0
        )
      end

      def create_optimizer
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []

        raise "Model is nil in create_optimizer" if @model.nil?

        @model.named_parameters.each do |name, param|
          next unless param.requires_grad

          if name.include?("bias") || name.include?("layernorm") || name.include?("norm")
            no_decay_params << param
          else
            decay_params << param
          end
        end

        # If no trainable params found (e.g., after LoRA), get LoRA params
        if decay_params.empty? && no_decay_params.empty?
          lora_params = LoRA.trainable_parameters(@model)
          if lora_params.any?
            decay_params = lora_params
          else
            raise TrainingError, "No trainable parameters found. Did you forget to apply LoRA or unfreeze layers?"
          end
        end

        param_groups = [
          { params: decay_params, weight_decay: @config.weight_decay },
          { params: no_decay_params, weight_decay: 0.0 }
        ].reject { |g| g[:params].empty? }

        Torch::Optim::AdamW.new(param_groups, lr: @config.learning_rate)
      end

      def create_scheduler(optimizer)
        return nil unless @config.warmup_steps && @config.warmup_steps > 0

        # Linear warmup then constant
        # Note: torch.rb scheduler support may be limited
        nil
      end

      def move_to_device(batch)
        batch.transform_values { |v| v.to(@device) }
      end

      # Manual gradient clipping implementation
      def clip_grad_norm(parameters, max_norm)
        total_norm = 0.0

        parameters.each do |param|
          next unless param.grad

          # Convert to float32 for .item (bfloat16 not supported)
          param_norm = param.grad.data.norm(2).to(:float32).item
          total_norm += param_norm ** 2
        end

        total_norm = Math.sqrt(total_norm)
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0
          parameters.each do |param|
            next unless param.grad

            param.grad.data.mul!(clip_coef)
          end
        end

        total_norm
      end
    end
  end
end
