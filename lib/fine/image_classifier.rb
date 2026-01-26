# frozen_string_literal: true

module Fine
  # High-level API for image classification fine-tuning
  #
  # @example Simple usage
  #   classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224")
  #   classifier.fit(train_dir: "data/train", epochs: 3)
  #   classifier.save("my_model")
  #
  # @example With configuration
  #   classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  #     config.learning_rate = 2e-4
  #     config.batch_size = 32
  #     config.on_epoch_end { |epoch, metrics| puts "Epoch #{epoch}: #{metrics}" }
  #   end
  #
  class ImageClassifier
    attr_reader :model, :config, :label_map, :model_id

    # Create a new ImageClassifier
    #
    # @param model_id [String] Hugging Face model ID (e.g., "google/siglip2-base-patch16-224")
    # @yield [config] Optional configuration block
    # @yieldparam config [Configuration] Configuration object
    def initialize(model_id, &block)
      @model_id = model_id
      @config = Configuration.new
      @model = nil
      @label_map = nil
      @trained = false

      # Apply configuration block
      block&.call(@config)

      # Add default progress bar if none configured
      if @config.callbacks.empty? && Fine.configuration&.progress_bar != false
        @config.callbacks << Callbacks::ProgressBar.new
      end
    end

    # Load a fine-tuned classifier from disk
    #
    # @param path [String] Path to saved model directory
    # @return [ImageClassifier]
    def self.load(path)
      config_path = File.join(path, "config.json")
      raise ModelNotFoundError.new(path, "Model not found") unless File.exist?(config_path)

      config_data = JSON.parse(File.read(config_path))

      classifier = allocate
      classifier.instance_variable_set(:@model_id, config_data["_model_id"] || "custom")
      classifier.instance_variable_set(:@config, Configuration.new)
      classifier.instance_variable_set(:@trained, true)

      # Load label map
      if config_data["label2id"]
        classifier.instance_variable_set(:@label_map, config_data["label2id"])
      elsif config_data["id2label"]
        classifier.instance_variable_set(
          :@label_map,
          config_data["id2label"].transform_keys(&:to_i).invert
        )
      end

      # Load model
      classifier.instance_variable_set(
        :@model,
        Models::SigLIP2ForImageClassification.load(path)
      )

      classifier
    end

    # Fine-tune the model on a dataset
    #
    # @param train_dir [String] Path to training data directory
    # @param val_dir [String, nil] Path to validation data directory
    # @param epochs [Integer, nil] Number of epochs (overrides config)
    # @return [Array<Hash>] Training history
    def fit(train_dir:, val_dir: nil, epochs: nil)
      # Override epochs if provided
      @config.epochs = epochs if epochs

      # Load datasets
      transforms = build_transforms
      train_dataset = Datasets::ImageDataset.from_directory(train_dir, transforms: transforms)

      val_dataset = if val_dir
        Datasets::ImageDataset.from_directory(val_dir, transforms: transforms)
      end

      # Store label map
      @label_map = train_dataset.label_map
      num_classes = train_dataset.num_classes

      # Load or create model
      @model = Models::SigLIP2ForImageClassification.from_pretrained(
        @model_id,
        num_labels: num_classes,
        freeze_encoder: @config.freeze_encoder,
        dropout: @config.dropout
      )

      # Create trainer and train
      trainer = Training::Trainer.new(
        @model,
        @config,
        train_dataset: train_dataset,
        val_dataset: val_dataset
      )

      history = trainer.fit
      @trained = true

      history
    end

    # Fine-tune with explicit train/val split
    #
    # @param data_dir [String] Path to data directory
    # @param val_split [Float] Fraction of data to use for validation
    # @param epochs [Integer, nil] Number of epochs
    # @return [Array<Hash>] Training history
    def fit_with_split(data_dir:, val_split: 0.2, epochs: nil)
      @config.epochs = epochs if epochs

      transforms = build_transforms
      full_dataset = Datasets::ImageDataset.from_directory(data_dir, transforms: transforms)
      train_dataset, val_dataset = full_dataset.split(test_size: val_split)

      @label_map = train_dataset.label_map
      num_classes = train_dataset.num_classes

      @model = Models::SigLIP2ForImageClassification.from_pretrained(
        @model_id,
        num_labels: num_classes,
        freeze_encoder: @config.freeze_encoder,
        dropout: @config.dropout
      )

      trainer = Training::Trainer.new(
        @model,
        @config,
        train_dataset: train_dataset,
        val_dataset: val_dataset
      )

      history = trainer.fit
      @trained = true

      history
    end

    # Make predictions on images
    #
    # @param images [String, Array<String>] Path(s) to image file(s)
    # @param top_k [Integer] Number of top predictions to return
    # @return [Array<Hash>] Predictions with :label and :score
    def predict(images, top_k: 5)
      raise TrainingError, "Model not trained or loaded" unless @trained && @model

      images = [images] if images.is_a?(String)
      transforms = build_inference_transforms

      # Load and transform images
      tensors = images.map do |path|
        image = Vips::Image.new_from_file(path, access: :sequential)
        transforms.call(image)
      end

      # Stack into batch
      pixel_values = Torch.stack(tensors)

      # Get predictions
      @model.eval
      probs = @model.predict_proba(pixel_values)

      # Convert to result format
      inverse_label_map = @label_map.invert

      probs.to_a.map do |sample_probs|
        # Get top-k predictions
        sorted = sample_probs.each_with_index.sort_by { |prob, _| -prob }
        top = sorted.first(top_k)

        top.map do |prob, idx|
          {
            label: inverse_label_map[idx] || idx.to_s,
            score: prob.round(4)
          }
        end
      end
    end

    # Save the fine-tuned model
    #
    # @param path [String] Directory path to save to
    def save(path)
      raise TrainingError, "Model not trained" unless @trained && @model

      @model.save(path, label_map: @label_map)

      # Also save the original model ID for reference
      config_path = File.join(path, "config.json")
      config = JSON.parse(File.read(config_path))
      config["_model_id"] = @model_id
      File.write(config_path, JSON.pretty_generate(config))
    end

    # Get class names in order of their IDs
    #
    # @return [Array<String>] Class names
    def class_names
      return [] unless @label_map

      @label_map.sort_by { |_, v| v }.map(&:first)
    end

    # Export to ONNX format
    #
    # @param path [String] Output path for ONNX file
    # @param options [Hash] Export options
    # @return [String] The output path
    def export_onnx(path, **options)
      Export.to_onnx(self, path, **options)
    end

    private

    def build_transforms
      transforms = []

      # Add augmentation transforms if configured
      if @config.augmentation_config.enabled?
        transforms.concat(@config.augmentation_config.to_transforms)
      end

      # Core transforms
      transforms << Transforms::Resize.new(@config.image_size)
      transforms << Transforms::ToTensor.new
      transforms << Transforms::Normalize.new

      Transforms::Compose.new(transforms)
    end

    def build_inference_transforms
      Transforms::Compose.new([
        Transforms::Resize.new(@config.image_size),
        Transforms::ToTensor.new,
        Transforms::Normalize.new
      ])
    end
  end
end
