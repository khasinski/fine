# frozen_string_literal: true

module Fine
  # High-level API for text classification
  #
  # @example Basic usage
  #   classifier = Fine::TextClassifier.new("distilbert-base-uncased")
  #   classifier.fit(train_file: "reviews.jsonl", epochs: 3)
  #   classifier.predict("This product is amazing!")
  #
  # @example With configuration
  #   classifier = Fine::TextClassifier.new("microsoft/deberta-v3-small") do |config|
  #     config.epochs = 5
  #     config.batch_size = 16
  #     config.learning_rate = 2e-5
  #   end
  #
  class TextClassifier
    attr_reader :model, :config, :tokenizer, :label_map, :model_id

    # Create a new TextClassifier
    #
    # @param model_id [String] HuggingFace model ID
    # @yield [config] Optional configuration block
    def initialize(model_id, &block)
      @model_id = model_id
      @config = TextConfiguration.new
      @model = nil
      @tokenizer = nil
      @label_map = nil
      @trained = false

      block&.call(@config)

      if @config.callbacks.empty? && Fine.configuration&.progress_bar != false
        @config.callbacks << Callbacks::ProgressBar.new
      end
    end

    # Load a fine-tuned classifier from disk
    #
    # @param path [String] Path to saved model directory
    # @return [TextClassifier]
    def self.load(path)
      config_path = File.join(path, "config.json")
      raise ModelNotFoundError.new(path) unless File.exist?(config_path)

      config_data = JSON.parse(File.read(config_path))

      classifier = allocate
      classifier.instance_variable_set(:@model_id, config_data["_model_id"] || "custom")
      classifier.instance_variable_set(:@config, TextConfiguration.new)
      classifier.instance_variable_set(:@trained, true)

      # Load label map
      if config_data["label2id"]
        classifier.instance_variable_set(:@label_map, config_data["label2id"])
      elsif config_data["id2label"]
        label_map = config_data["id2label"].transform_keys(&:to_i).invert
        classifier.instance_variable_set(:@label_map, label_map)
      end

      # Load tokenizer
      tokenizer_path = File.join(path, "tokenizer.json")
      tokenizer = if File.exist?(tokenizer_path)
        Tokenizers::AutoTokenizer.new(path, max_length: config_data["max_length"] || 512)
      else
        Tokenizers::AutoTokenizer.from_pretrained(
          config_data["_model_id"] || "distilbert-base-uncased",
          max_length: config_data["max_length"] || 512
        )
      end
      classifier.instance_variable_set(:@tokenizer, tokenizer)

      # Load model
      classifier.instance_variable_set(
        :@model,
        Models::BertForSequenceClassification.load(path)
      )

      classifier
    end

    # Fine-tune on a dataset
    #
    # @param train_file [String] Path to training data (JSONL or CSV)
    # @param val_file [String, nil] Path to validation data
    # @param epochs [Integer, nil] Override config epochs
    # @return [Array<Hash>] Training history
    def fit(train_file:, val_file: nil, epochs: nil)
      @config.epochs = epochs if epochs

      # Load tokenizer
      @tokenizer = Tokenizers::AutoTokenizer.from_pretrained(
        @model_id,
        max_length: @config.max_length
      )

      # Load datasets
      train_dataset = Datasets::TextDataset.from_file(train_file, tokenizer: @tokenizer)

      val_dataset = if val_file
        Datasets::TextDataset.from_file(val_file, tokenizer: @tokenizer, label_map: train_dataset.label_map)
      end

      @label_map = train_dataset.label_map
      num_classes = train_dataset.num_classes

      # Load model
      @model = Models::BertForSequenceClassification.from_pretrained(
        @model_id,
        num_labels: num_classes,
        dropout: @config.dropout
      )

      # Train
      trainer = Training::TextTrainer.new(
        @model,
        @config,
        train_dataset: train_dataset,
        val_dataset: val_dataset
      )

      history = trainer.fit
      @trained = true

      history
    end

    # Fine-tune with automatic train/val split
    #
    # @param data_file [String] Path to data file
    # @param val_split [Float] Fraction for validation
    # @return [Array<Hash>] Training history
    def fit_with_split(data_file:, val_split: 0.2, epochs: nil)
      @config.epochs = epochs if epochs

      @tokenizer = Tokenizers::AutoTokenizer.from_pretrained(
        @model_id,
        max_length: @config.max_length
      )

      full_dataset = Datasets::TextDataset.from_file(data_file, tokenizer: @tokenizer)
      train_dataset, val_dataset = full_dataset.split(test_size: val_split)

      @label_map = train_dataset.label_map

      @model = Models::BertForSequenceClassification.from_pretrained(
        @model_id,
        num_labels: train_dataset.num_classes,
        dropout: @config.dropout
      )

      trainer = Training::TextTrainer.new(
        @model,
        @config,
        train_dataset: train_dataset,
        val_dataset: val_dataset
      )

      history = trainer.fit
      @trained = true

      history
    end

    # Make predictions
    #
    # @param texts [String, Array<String>] Text(s) to classify
    # @param top_k [Integer] Number of top predictions to return
    # @return [Array<Array<Hash>>] Predictions with :label and :score
    def predict(texts, top_k: 5)
      raise TrainingError, "Model not trained or loaded" unless @trained && @model

      texts = [texts] if texts.is_a?(String)

      # Tokenize
      encoding = @tokenizer.encode(texts)

      # Get predictions
      @model.eval
      probs = @model.predict_proba(
        encoding[:input_ids],
        attention_mask: encoding[:attention_mask],
        token_type_ids: encoding[:token_type_ids]
      )

      # Convert to result format
      inverse_label_map = @label_map.invert

      probs.to_a.map do |sample_probs|
        sorted = sample_probs.each_with_index.sort_by { |prob, _| -prob }
        top = sorted.first([top_k, @label_map.size].min)

        top.map do |prob, idx|
          {
            label: inverse_label_map[idx] || idx.to_s,
            score: prob.round(4)
          }
        end
      end
    end

    # Save the model
    #
    # @param path [String] Directory to save to
    def save(path)
      raise TrainingError, "Model not trained" unless @trained && @model

      @model.save(path, label_map: @label_map)
      @tokenizer.save(path)

      # Update config with model ID and max_length
      config_path = File.join(path, "config.json")
      config = JSON.parse(File.read(config_path))
      config["_model_id"] = @model_id
      config["max_length"] = @config.max_length
      File.write(config_path, JSON.pretty_generate(config))
    end

    # Get class names
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
  end

  # Configuration for text models
  class TextConfiguration < Configuration
    attr_accessor :max_length, :warmup_ratio

    def initialize
      super
      @max_length = 256
      @warmup_ratio = 0.1
      @learning_rate = 2e-5  # Lower default for text models
      @batch_size = 16       # Smaller default for text
    end
  end
end
