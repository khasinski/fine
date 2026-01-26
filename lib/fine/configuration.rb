# frozen_string_literal: true

module Fine
  # Configuration for training runs
  class Configuration
    # Training hyperparameters
    attr_accessor :epochs, :batch_size, :learning_rate, :weight_decay
    attr_accessor :warmup_steps, :warmup_ratio
    attr_accessor :optimizer, :scheduler

    # Model configuration
    attr_accessor :freeze_encoder, :dropout, :num_labels

    # Data configuration
    attr_accessor :image_size

    # Callbacks
    attr_accessor :callbacks

    # Augmentation
    attr_reader :augmentation_config

    def initialize
      # Training defaults
      @epochs = 3
      @batch_size = 32
      @learning_rate = 2e-4
      @weight_decay = 0.02
      @warmup_steps = 0
      @warmup_ratio = 0.0
      @optimizer = :adamw
      @scheduler = :cosine

      # Model defaults
      @freeze_encoder = false
      @dropout = 0.1
      @num_labels = nil # auto-detect from dataset

      # Data defaults
      @image_size = 224

      # Callbacks
      @callbacks = []

      # Augmentation
      @augmentation_config = AugmentationConfig.new
    end

    def augmentation
      yield @augmentation_config if block_given?
      @augmentation_config
    end

    # Register a callback for epoch end
    def on_epoch_end(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_epoch_end: block)
    end

    # Register a callback for batch end
    def on_batch_end(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_batch_end: block)
    end

    # Register a callback for train begin
    def on_train_begin(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_train_begin: block)
    end

    # Register a callback for train end
    def on_train_end(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_train_end: block)
    end
  end

  # Configuration for data augmentation
  class AugmentationConfig
    attr_accessor :random_horizontal_flip, :random_vertical_flip
    attr_accessor :random_rotation, :color_jitter
    attr_accessor :random_resized_crop

    def initialize
      @random_horizontal_flip = false
      @random_vertical_flip = false
      @random_rotation = 0
      @color_jitter = nil
      @random_resized_crop = nil
    end

    def enabled?
      @random_horizontal_flip ||
        @random_vertical_flip ||
        @random_rotation.positive? ||
        @color_jitter ||
        @random_resized_crop
    end

    def to_transforms
      transforms = []
      transforms << Transforms::RandomHorizontalFlip.new if @random_horizontal_flip
      transforms << Transforms::RandomVerticalFlip.new if @random_vertical_flip
      transforms << Transforms::RandomRotation.new(@random_rotation) if @random_rotation.positive?
      # Add more transforms as implemented
      transforms
    end
  end
end
