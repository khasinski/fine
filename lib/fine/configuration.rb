# frozen_string_literal: true

module Fine
  # Configuration for training runs
  #
  # @example Basic usage
  #   Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  #     config.epochs = 5
  #     config.batch_size = 16
  #   end
  #
  # @example With callbacks
  #   config.on_epoch_end do |epoch, metrics|
  #     puts "Epoch #{epoch}: loss=#{metrics[:loss]}"
  #   end
  #
  class Configuration
    # Default values for all configurations
    DEFAULTS = {
      epochs: 3,
      batch_size: 16,
      learning_rate: 2e-5,
      weight_decay: 0.01,
      warmup_ratio: 0.1,
      optimizer: :adamw,
      scheduler: :linear,
      dropout: 0.1,
      image_size: 224
    }.freeze

    # Training hyperparameters
    # @!attribute epochs
    #   @return [Integer] Number of training epochs (default: 3)
    # @!attribute batch_size
    #   @return [Integer] Samples per batch (default: 16)
    # @!attribute learning_rate
    #   @return [Float] Learning rate (default: 2e-5)
    # @!attribute weight_decay
    #   @return [Float] L2 regularization (default: 0.01)
    attr_accessor :epochs, :batch_size, :learning_rate, :weight_decay

    # @!attribute warmup_steps
    #   @return [Integer] Number of warmup steps (default: 0, use warmup_ratio instead)
    # @!attribute warmup_ratio
    #   @return [Float] Fraction of training for warmup (default: 0.1)
    attr_accessor :warmup_steps, :warmup_ratio

    # @!attribute optimizer
    #   @return [Symbol] Optimizer type (:adamw, :adam, :sgd) (default: :adamw)
    # @!attribute scheduler
    #   @return [Symbol] LR scheduler (:linear, :cosine, :constant) (default: :linear)
    attr_accessor :optimizer, :scheduler

    # Model configuration
    # @!attribute freeze_encoder
    #   @return [Boolean] Freeze encoder weights, only train head (default: false)
    # @!attribute dropout
    #   @return [Float] Dropout probability (default: 0.1)
    # @!attribute num_labels
    #   @return [Integer, nil] Number of output classes (auto-detected if nil)
    attr_accessor :freeze_encoder, :dropout, :num_labels

    # Data configuration
    # @!attribute image_size
    #   @return [Integer] Target image size for resizing (default: 224)
    attr_accessor :image_size

    # @!attribute callbacks
    #   @return [Array<Callbacks::Base>] Training callbacks
    attr_accessor :callbacks

    # @!attribute augmentation_config
    #   @return [AugmentationConfig] Data augmentation settings
    attr_reader :augmentation_config

    def initialize
      # Training defaults - optimized for most tasks
      @epochs = DEFAULTS[:epochs]
      @batch_size = DEFAULTS[:batch_size]
      @learning_rate = DEFAULTS[:learning_rate]
      @weight_decay = DEFAULTS[:weight_decay]
      @warmup_steps = 0
      @warmup_ratio = DEFAULTS[:warmup_ratio]
      @optimizer = DEFAULTS[:optimizer]
      @scheduler = DEFAULTS[:scheduler]

      # Model defaults
      @freeze_encoder = false
      @dropout = DEFAULTS[:dropout]
      @num_labels = nil # auto-detect from dataset

      # Data defaults
      @image_size = DEFAULTS[:image_size]

      # Callbacks
      @callbacks = []

      # Augmentation
      @augmentation_config = AugmentationConfig.new
    end

    # Configure data augmentation
    #
    # @yield [AugmentationConfig] The augmentation configuration
    # @return [AugmentationConfig]
    #
    # @example
    #   config.augmentation do |aug|
    #     aug.random_horizontal_flip = true
    #     aug.random_rotation = 15
    #   end
    def augmentation
      yield @augmentation_config if block_given?
      @augmentation_config
    end

    # Register a callback for epoch end
    #
    # @yield [Integer, Hash] Epoch number and metrics hash
    #
    # @example
    #   config.on_epoch_end do |epoch, metrics|
    #     puts "Epoch #{epoch}: loss=#{metrics[:loss]}"
    #   end
    def on_epoch_end(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_epoch_end: block)
    end

    # Register a callback for batch end
    #
    # @yield [Integer, Float] Batch index and loss value
    def on_batch_end(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_batch_end: block)
    end

    # Register a callback for training start
    #
    # @yield [Hash] Training info (model, config)
    def on_train_begin(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_train_begin: block)
    end

    # Register a callback for training end
    #
    # @yield [Array<Hash>] Training history
    def on_train_end(&block)
      @callbacks << Callbacks::LambdaCallback.new(on_train_end: block)
    end

    # Return configuration as a hash
    def to_h
      {
        epochs: @epochs,
        batch_size: @batch_size,
        learning_rate: @learning_rate,
        weight_decay: @weight_decay,
        warmup_steps: @warmup_steps,
        warmup_ratio: @warmup_ratio,
        optimizer: @optimizer,
        scheduler: @scheduler,
        freeze_encoder: @freeze_encoder,
        dropout: @dropout,
        num_labels: @num_labels,
        image_size: @image_size
      }
    end
  end

  # Configuration for text models (BERT, DistilBERT, DeBERTa)
  class TextConfiguration < Configuration
    # @!attribute max_length
    #   @return [Integer] Maximum sequence length (default: 128)
    attr_accessor :max_length

    # Text model defaults
    DEFAULTS = Configuration::DEFAULTS.merge(
      max_length: 128,
      batch_size: 16
    ).freeze

    def initialize
      super
      @max_length = DEFAULTS[:max_length]
      @batch_size = DEFAULTS[:batch_size]
    end
  end

  # Configuration for embedding models (Sentence Transformers)
  class EmbeddingConfiguration < Configuration
    # @!attribute max_length
    #   @return [Integer] Maximum sequence length (default: 128)
    # @!attribute pooling_mode
    #   @return [Symbol] Pooling strategy (:mean, :cls, :max) (default: :mean)
    # @!attribute loss
    #   @return [Symbol] Loss function (:cosine, :contrastive, :triplet) (default: :cosine)
    attr_accessor :max_length, :pooling_mode, :loss

    # Embedding model defaults
    DEFAULTS = Configuration::DEFAULTS.merge(
      max_length: 128,
      pooling_mode: :mean,
      loss: :cosine,
      batch_size: 32
    ).freeze

    def initialize
      super
      @max_length = DEFAULTS[:max_length]
      @pooling_mode = DEFAULTS[:pooling_mode]
      @loss = DEFAULTS[:loss]
      @batch_size = DEFAULTS[:batch_size]
    end
  end

  # Configuration for data augmentation
  #
  # @example
  #   config.augmentation do |aug|
  #     aug.random_horizontal_flip = true
  #     aug.random_rotation = 15
  #     aug.color_jitter = { brightness: 0.2, contrast: 0.2 }
  #   end
  class AugmentationConfig
    # @!attribute random_horizontal_flip
    #   @return [Boolean] Randomly flip images horizontally (default: false)
    # @!attribute random_vertical_flip
    #   @return [Boolean] Randomly flip images vertically (default: false)
    # @!attribute random_rotation
    #   @return [Integer] Max rotation degrees (0 = disabled) (default: 0)
    # @!attribute color_jitter
    #   @return [Hash, nil] Color jitter settings { brightness:, contrast:, saturation:, hue: }
    # @!attribute random_resized_crop
    #   @return [Hash, nil] Random crop settings { scale:, ratio: }
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

    # Check if any augmentation is enabled
    def enabled?
      @random_horizontal_flip ||
        @random_vertical_flip ||
        @random_rotation.positive? ||
        @color_jitter ||
        @random_resized_crop
    end

    # Convert to transform objects
    def to_transforms
      transforms = []
      transforms << Transforms::RandomHorizontalFlip.new if @random_horizontal_flip
      transforms << Transforms::RandomVerticalFlip.new if @random_vertical_flip
      transforms << Transforms::RandomRotation.new(@random_rotation) if @random_rotation.positive?
      transforms
    end
  end
end
