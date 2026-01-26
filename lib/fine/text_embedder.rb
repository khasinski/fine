# frozen_string_literal: true

module Fine
  # High-level API for text embeddings
  #
  # @example Basic usage
  #   embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2")
  #   embedder.fit(train_file: "pairs.jsonl", epochs: 3)
  #   embedding = embedder.encode("Hello world")
  #
  # @example Without fine-tuning (use pretrained directly)
  #   embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2")
  #   embedding = embedder.encode("Hello world")
  #
  class TextEmbedder
    attr_reader :model, :config, :tokenizer, :model_id

    # Create a new TextEmbedder
    #
    # @param model_id [String] HuggingFace model ID
    # @yield [config] Optional configuration block
    def initialize(model_id, &block)
      @model_id = model_id
      @config = EmbeddingConfiguration.new
      @model = nil
      @tokenizer = nil
      @trained = false

      block&.call(@config)

      # Load tokenizer immediately for encoding
      @tokenizer = Tokenizers::AutoTokenizer.from_pretrained(
        model_id,
        max_length: @config.max_length
      )

      # Load pretrained model for immediate use
      @model = Models::SentenceTransformer.from_pretrained(
        model_id,
        pooling_mode: @config.pooling_mode
      )
      @trained = true  # Pretrained is ready to use
    end

    # Load a fine-tuned embedder from disk
    #
    # @param path [String] Path to saved model
    # @return [TextEmbedder]
    def self.load(path)
      config_path = File.join(path, "config.json")
      raise ModelNotFoundError.new(path) unless File.exist?(config_path)

      config_data = JSON.parse(File.read(config_path))

      embedder = allocate
      embedder.instance_variable_set(:@model_id, config_data["_model_id"] || "custom")
      embedder.instance_variable_set(:@config, EmbeddingConfiguration.new)
      embedder.instance_variable_set(:@trained, true)

      # Load tokenizer
      tokenizer_path = File.join(path, "tokenizer.json")
      tokenizer = if File.exist?(tokenizer_path)
        Tokenizers::AutoTokenizer.new(path, max_length: config_data["max_length"] || 512)
      else
        Tokenizers::AutoTokenizer.from_pretrained(
          config_data["_model_id"],
          max_length: config_data["max_length"] || 512
        )
      end
      embedder.instance_variable_set(:@tokenizer, tokenizer)

      # Load model
      embedder.instance_variable_set(
        :@model,
        Models::SentenceTransformer.load(path)
      )

      embedder
    end

    # Fine-tune on pairs/triplets data
    #
    # @param train_file [String] Path to training data (JSONL)
    # @param epochs [Integer, nil] Override config epochs
    # @return [Array<Hash>] Training history
    def fit(train_file:, epochs: nil)
      @config.epochs = epochs if epochs

      # Load dataset
      train_dataset = Datasets::TextPairDataset.from_jsonl(
        train_file,
        tokenizer: @tokenizer,
        text_a_column: "query",
        text_b_column: "positive"
      )

      # Add progress bar callback
      if @config.callbacks.empty? && Fine.configuration&.progress_bar != false
        @config.callbacks << Callbacks::ProgressBar.new
      end

      # Train
      trainer = Training::EmbeddingTrainer.new(
        @model,
        @config,
        train_dataset: train_dataset
      )

      trainer.fit
    end

    # Encode text(s) to embeddings
    #
    # @param texts [String, Array<String>] Text(s) to encode
    # @return [Array<Float>, Array<Array<Float>>] Embedding(s)
    def encode(texts)
      raise TrainingError, "Model not loaded" unless @model

      single_input = texts.is_a?(String)
      texts = [texts] if single_input

      # Tokenize
      encoding = @tokenizer.encode(texts)

      # Get embeddings
      embeddings = @model.encode(
        encoding[:input_ids],
        attention_mask: encoding[:attention_mask]
      )

      # Convert to Ruby arrays
      result = embeddings.to_a

      single_input ? result.first : result
    end

    # Compute similarity between two texts
    #
    # @param text_a [String] First text
    # @param text_b [String] Second text
    # @return [Float] Cosine similarity score
    def similarity(text_a, text_b)
      emb_a = encode(text_a)
      emb_b = encode(text_b)

      cosine_similarity(emb_a, emb_b)
    end

    # Find most similar texts from a corpus
    #
    # @param query [String] Query text
    # @param corpus [Array<String>] Corpus to search
    # @param top_k [Integer] Number of results
    # @return [Array<Hash>] Results with :text, :score, :index
    def search(query, corpus, top_k: 5)
      query_emb = encode(query)
      corpus_embs = encode(corpus)

      scores = corpus_embs.map.with_index do |emb, idx|
        { text: corpus[idx], score: cosine_similarity(query_emb, emb), index: idx }
      end

      scores.sort_by { |s| -s[:score] }.first(top_k)
    end

    # Save the model
    #
    # @param path [String] Directory to save to
    def save(path)
      raise TrainingError, "Model not loaded" unless @model

      @model.save(path)
      @tokenizer.save(path)

      # Update config
      config_path = File.join(path, "config.json")
      config = JSON.parse(File.read(config_path))
      config["_model_id"] = @model_id
      config["max_length"] = @config.max_length
      File.write(config_path, JSON.pretty_generate(config))
    end

    # Get embedding dimension
    def embedding_dim
      @model.config.hidden_size
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

    def cosine_similarity(a, b)
      dot = a.zip(b).sum { |x, y| x * y }
      norm_a = Math.sqrt(a.sum { |x| x * x })
      norm_b = Math.sqrt(b.sum { |x| x * x })
      dot / (norm_a * norm_b)
    end
  end

  # Configuration for embedding models
  class EmbeddingConfiguration < Configuration
    attr_accessor :max_length, :pooling_mode, :loss

    def initialize
      super
      @max_length = 256
      @pooling_mode = :mean
      @loss = :multiple_negatives_ranking
      @learning_rate = 2e-5
      @batch_size = 32
      @epochs = 1
    end
  end
end
