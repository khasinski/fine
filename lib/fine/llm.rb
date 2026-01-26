# frozen_string_literal: true

module Fine
  # High-level API for LLM fine-tuning
  #
  # @example Basic fine-tuning
  #   llm = Fine::LLM.new("meta-llama/Llama-3.2-1B")
  #   llm.fit(train_file: "instructions.jsonl", epochs: 3)
  #   llm.save("my_llama")
  #
  # @example Generation
  #   llm = Fine::LLM.load("my_llama")
  #   response = llm.generate("What is Ruby?", max_new_tokens: 100)
  #
  # @example With configuration
  #   llm = Fine::LLM.new("google/gemma-2b") do |config|
  #     config.epochs = 3
  #     config.batch_size = 4
  #     config.learning_rate = 1e-5
  #     config.max_length = 1024
  #   end
  #
  class LLM
    attr_reader :model, :config, :tokenizer, :model_id

    # Create a new LLM for fine-tuning
    #
    # @param model_id [String] HuggingFace model ID
    # @yield [config] Optional configuration block
    def initialize(model_id, &block)
      @model_id = model_id
      @config = LLMConfiguration.new
      @model = nil
      @tokenizer = nil
      @trained = false

      block&.call(@config)

      if @config.callbacks.empty? && Fine.configuration&.progress_bar != false
        @config.callbacks << Callbacks::ProgressBar.new
      end
    end

    # Load a fine-tuned LLM from disk
    #
    # @param path [String] Path to saved model
    # @return [LLM]
    def self.load(path)
      config_path = File.join(path, "config.json")
      raise ModelNotFoundError.new(path) unless File.exist?(config_path)

      config_data = JSON.parse(File.read(config_path))

      llm = allocate
      llm.instance_variable_set(:@model_id, config_data["_model_id"] || "custom")
      llm.instance_variable_set(:@config, LLMConfiguration.new)
      llm.instance_variable_set(:@trained, true)

      # Load tokenizer
      tokenizer_path = File.join(path, "tokenizer.json")
      tokenizer = if File.exist?(tokenizer_path)
        Tokenizers::AutoTokenizer.new(path, max_length: config_data["max_length"] || 2048)
      else
        Tokenizers::AutoTokenizer.from_pretrained(
          config_data["_model_id"],
          max_length: config_data["max_length"] || 2048
        )
      end
      llm.instance_variable_set(:@tokenizer, tokenizer)

      # Load model
      llm.instance_variable_set(
        :@model,
        Models::CausalLM.load(path)
      )

      llm
    end

    # Fine-tune on instruction data
    #
    # @param train_file [String] Path to training data (JSONL)
    # @param val_file [String, nil] Path to validation data
    # @param format [Symbol] Data format (:alpaca, :sharegpt, :simple, :auto)
    # @param epochs [Integer, nil] Override config epochs
    # @return [Array<Hash>] Training history
    def fit(train_file:, val_file: nil, format: :auto, epochs: nil)
      @config.epochs = epochs if epochs

      # Download model files first (including tokenizer)
      downloader = Hub::ModelDownloader.new(@model_id)
      model_path = downloader.download

      # Load tokenizer from cache or HuggingFace
      tokenizer_path = File.join(model_path, "tokenizer.json")
      @tokenizer = if File.exist?(tokenizer_path)
        Tokenizers::AutoTokenizer.new(model_path, max_length: @config.max_length)
      else
        Tokenizers::AutoTokenizer.from_pretrained(@model_id, max_length: @config.max_length)
      end

      # Set pad token if not set
      @config.pad_token_id ||= @tokenizer.pad_token_id || @tokenizer.eos_token_id || 0

      # Load datasets
      train_dataset = Datasets::InstructionDataset.from_jsonl(
        train_file,
        tokenizer: @tokenizer,
        format: format,
        max_length: @config.max_length
      )

      val_dataset = if val_file
        Datasets::InstructionDataset.from_jsonl(
          val_file,
          tokenizer: @tokenizer,
          format: format,
          max_length: @config.max_length
        )
      end

      # Load model
      @model = Models::CausalLM.from_pretrained(@model_id)

      # Freeze layers if configured
      if @config.freeze_layers && @config.freeze_layers > 0
        freeze_bottom_layers(@config.freeze_layers)
      end

      # Train
      trainer = Training::LLMTrainer.new(
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
    # @param format [Symbol] Data format
    # @return [Array<Hash>] Training history
    def fit_with_split(data_file:, val_split: 0.1, format: :auto, epochs: nil)
      @config.epochs = epochs if epochs

      # Download model files first (including tokenizer)
      downloader = Hub::ModelDownloader.new(@model_id)
      model_path = downloader.download

      # Load tokenizer from cache or HuggingFace
      tokenizer_path = File.join(model_path, "tokenizer.json")
      @tokenizer = if File.exist?(tokenizer_path)
        Tokenizers::AutoTokenizer.new(model_path, max_length: @config.max_length)
      else
        Tokenizers::AutoTokenizer.from_pretrained(@model_id, max_length: @config.max_length)
      end

      @config.pad_token_id ||= @tokenizer.pad_token_id || @tokenizer.eos_token_id || 0

      full_dataset = Datasets::InstructionDataset.from_jsonl(
        data_file,
        tokenizer: @tokenizer,
        format: format,
        max_length: @config.max_length
      )

      train_dataset, val_dataset = full_dataset.split(test_size: val_split)

      @model = Models::CausalLM.from_pretrained(@model_id)

      if @config.freeze_layers && @config.freeze_layers > 0
        freeze_bottom_layers(@config.freeze_layers)
      end

      trainer = Training::LLMTrainer.new(
        @model,
        @config,
        train_dataset: train_dataset,
        val_dataset: val_dataset
      )

      history = trainer.fit
      @trained = true

      history
    end

    # Generate text
    #
    # @param prompt [String] Input prompt
    # @param max_new_tokens [Integer] Maximum tokens to generate
    # @param temperature [Float] Sampling temperature (higher = more random)
    # @param top_p [Float] Nucleus sampling threshold
    # @param top_k [Integer] Top-k sampling
    # @param do_sample [Boolean] Whether to sample (false = greedy)
    # @return [String] Generated text
    def generate(prompt, max_new_tokens: 100, temperature: 0.7, top_p: 0.9, top_k: 50, do_sample: true)
      raise TrainingError, "Model not loaded" unless @model && @tokenizer

      # Tokenize prompt without padding for autoregressive generation
      ids = @tokenizer.encode_for_generation(prompt)
      input_ids = Torch.tensor([ids])

      # Move to device
      input_ids = input_ids.to(Fine.device)
      @model.to(Fine.device)

      # Generate
      output_ids = @model.generate(
        input_ids,
        max_new_tokens: max_new_tokens,
        temperature: temperature,
        top_p: top_p,
        top_k: top_k,
        do_sample: do_sample,
        eos_token_id: @tokenizer.eos_token_id,
        pad_token_id: @config.pad_token_id
      )

      # Decode
      @tokenizer.decode(output_ids[0].to_a)
    end

    # Chat-style generation
    #
    # @param messages [Array<Hash>] Messages with :role and :content
    # @param kwargs [Hash] Generation parameters
    # @return [String] Assistant response
    def chat(messages, **kwargs)
      prompt = format_chat_prompt(messages)
      full_response = generate(prompt, **kwargs)

      # Extract just the assistant response
      if full_response.include?("### Response:")
        full_response.split("### Response:").last.strip
      else
        # Remove the prompt from the response
        full_response[prompt.length..].strip
      end
    end

    # Save the model
    #
    # @param path [String] Directory to save to
    def save(path)
      raise TrainingError, "Model not trained or loaded" unless @model

      @model.save(path)
      @tokenizer.save(path)

      # Update config with model ID
      config_path = File.join(path, "config.json")
      config = JSON.parse(File.read(config_path))
      config["_model_id"] = @model_id
      config["max_length"] = @config.max_length
      File.write(config_path, JSON.pretty_generate(config))
    end

    # Export to GGUF format for llama.cpp, ollama, etc.
    #
    # @param path [String] Output path for GGUF file
    # @param quantization [Symbol] Quantization type (:f16, :q4_0, :q8_0, etc.)
    # @param metadata [Hash] Additional metadata
    # @return [String] The output path
    def export_gguf(path, quantization: :f16, **options)
      Export.to_gguf(self, path, quantization: quantization, **options)
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

    def freeze_bottom_layers(num_layers)
      # Freeze embedding
      @model.decoder.embed_tokens.parameters.each { |p| p.requires_grad = false }

      # Freeze bottom N layers
      @model.decoder.layers[0...num_layers].each do |layer|
        layer.parameters.each { |p| p.requires_grad = false }
      end
    end

    def format_chat_prompt(messages)
      prompt = ""

      messages.each do |msg|
        case msg[:role]
        when "system"
          prompt += "### System:\n#{msg[:content]}\n\n"
        when "user"
          prompt += "### Instruction:\n#{msg[:content]}\n\n"
        when "assistant"
          prompt += "### Response:\n#{msg[:content]}\n\n"
        end
      end

      # Add response prefix for the model to continue
      prompt += "### Response:\n" unless prompt.end_with?("### Response:\n")

      prompt
    end
  end

  # Configuration for LLM fine-tuning
  #
  # @example
  #   llm = Fine::LLM.new("google/gemma-3-1b-it") do |config|
  #     config.epochs = 3
  #     config.max_length = 512
  #     config.learning_rate = 1e-5
  #   end
  #
  class LLMConfiguration < Configuration
    # LLM-specific defaults
    DEFAULTS = Configuration::DEFAULTS.merge(
      max_length: 2048,
      learning_rate: 2e-5,
      batch_size: 4,
      epochs: 3,
      warmup_steps: 100,
      gradient_accumulation_steps: 4,
      max_grad_norm: 1.0
    ).freeze

    # @!attribute max_length
    #   @return [Integer] Maximum sequence length (default: 2048)
    # @!attribute gradient_accumulation_steps
    #   @return [Integer] Accumulate gradients over N steps (default: 4)
    # @!attribute max_grad_norm
    #   @return [Float] Gradient clipping norm (default: 1.0)
    # @!attribute freeze_layers
    #   @return [Integer] Number of bottom layers to freeze (default: 0)
    # @!attribute pad_token_id
    #   @return [Integer, nil] Padding token ID (auto-detected if nil)
    attr_accessor :max_length, :warmup_steps, :gradient_accumulation_steps,
                  :max_grad_norm, :freeze_layers, :pad_token_id

    def initialize
      super
      @max_length = DEFAULTS[:max_length]
      @learning_rate = DEFAULTS[:learning_rate]
      @batch_size = DEFAULTS[:batch_size]
      @epochs = DEFAULTS[:epochs]
      @warmup_steps = DEFAULTS[:warmup_steps]
      @gradient_accumulation_steps = DEFAULTS[:gradient_accumulation_steps]
      @max_grad_norm = DEFAULTS[:max_grad_norm]
      @freeze_layers = 0
      @pad_token_id = nil
    end
  end
end
