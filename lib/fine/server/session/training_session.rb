# frozen_string_literal: true

require "securerandom"

module Fine
  module Server
    module Session
      # Manages a single training session
      class TrainingSession
        TYPES = %i[llm text_classifier image_classifier embedder].freeze
        STATUSES = %i[pending running completed failed cancelled].freeze

        attr_reader :id, :type, :status, :config, :metrics, :history, :error, :started_at, :completed_at

        def initialize(type:, config:)
          @id = SecureRandom.uuid
          @type = type
          @config = config
          @status = :pending
          @metrics = {}
          @history = []
          @error = nil
          @model = nil
          @thread = nil
          @subscribers = []
          @mutex = Mutex.new
          @started_at = nil
          @completed_at = nil
        end

        def start
          @status = :running
          @started_at = Time.now
          @thread = Thread.new { run_training }
        end

        def cancel
          @status = :cancelled
          @completed_at = Time.now
          @thread&.kill
          broadcast_event(:training_end, { status: :cancelled })
        end

        def subscribe(&block)
          @mutex.synchronize { @subscribers << block }
        end

        def unsubscribe(block)
          @mutex.synchronize { @subscribers.delete(block) }
        end

        def broadcast_event(event, data)
          @mutex.synchronize do
            @subscribers.each { |sub| sub.call(event, data) }
          end
        end

        def to_json
          {
            id: @id,
            type: @type,
            status: @status,
            config: @config,
            metrics: @metrics,
            history: @history,
            error: @error,
            started_at: @started_at&.iso8601,
            completed_at: @completed_at&.iso8601
          }
        end

        # Chat with trained LLM
        def chat(message)
          return "Model not ready" unless @model && @status == :completed

          @model.generate(message)
        end

        # Classify text or image
        def classify(input)
          return [] unless @model && @status == :completed

          @model.predict(input)
        end

        # Similarity search with embeddings
        def similarity_search(query, corpus, top_k: 5)
          return [] unless @model && @status == :completed

          @model.search(query, corpus, top_k: top_k)
        end

        # Export to ONNX
        def export_onnx
          return nil unless @model && @status == :completed

          exports_dir = File.join(Dir.pwd, "exports")
          FileUtils.mkdir_p(exports_dir)
          filename = File.join(exports_dir, "#{@id[0..7]}.onnx")
          @model.export_onnx(filename)
          filename
        end

        # Export to GGUF (LLMs only)
        def export_gguf(quantization: :f16)
          return nil unless @model && @status == :completed && @type == :llm

          exports_dir = File.join(Dir.pwd, "exports")
          FileUtils.mkdir_p(exports_dir)
          filename = File.join(exports_dir, "#{@id[0..7]}.gguf")
          @model.export_gguf(filename, quantization: quantization)
          filename
        end

        # Save model
        def save_model(name)
          return nil unless @model && @status == :completed

          path = File.join(Dir.pwd, "models", name)
          @model.save(path)
          path
        end

        private

        def run_training
          case @type
          when :llm
            train_llm
          when :text_classifier
            train_text_classifier
          when :image_classifier
            train_image_classifier
          when :embedder
            train_embedder
          end
          @status = :completed
          @completed_at = Time.now
          broadcast_event(:training_end, { status: :completed, history: @history })
        rescue StandardError => e
          @error = e.message
          @status = :failed
          @completed_at = Time.now
          broadcast_event(:training_end, { status: :failed, error: @error })
        end

        def train_llm
          model_id = @config[:model_id] || "google/gemma-3-1b-it"

          @model = Fine::LLM.new(model_id) do |cfg|
            cfg.epochs = @config[:epochs] if @config[:epochs]
            cfg.batch_size = @config[:batch_size] if @config[:batch_size]
            cfg.learning_rate = @config[:learning_rate] if @config[:learning_rate]
            cfg.max_length = @config[:max_length] if @config[:max_length]
            cfg.callbacks << Callbacks::WebCallback.new(self)

            # LoRA configuration
            if @config[:use_lora]
              cfg.use_lora = true
              cfg.lora_rank = @config[:lora_rank] || 8
              cfg.lora_alpha = @config[:lora_alpha] || 16
            end
          end

          @model.fit(train_file: @config[:train_file], val_file: @config[:val_file])
        end

        def train_text_classifier
          model_id = @config[:model_id] || "distilbert-base-uncased"

          @model = Fine::TextClassifier.new(model_id) do |cfg|
            cfg.epochs = @config[:epochs]
            cfg.batch_size = @config[:batch_size]
            cfg.learning_rate = @config[:learning_rate]
            cfg.max_length = @config[:max_length]
            cfg.callbacks << Callbacks::WebCallback.new(self)
          end

          @model.fit(train_file: @config[:train_file], val_file: @config[:val_file])
        end

        def train_image_classifier
          model_id = @config[:model_id] || "google/siglip2-base-patch16-224"

          @model = Fine::ImageClassifier.new(model_id) do |cfg|
            cfg.epochs = @config[:epochs]
            cfg.batch_size = @config[:batch_size]
            cfg.learning_rate = @config[:learning_rate]
            cfg.callbacks << Callbacks::WebCallback.new(self)
          end

          @model.fit(train_dir: @config[:train_dir], val_dir: @config[:val_dir])
        end

        def train_embedder
          model_id = @config[:model_id] || "sentence-transformers/all-MiniLM-L6-v2"

          @model = Fine::TextEmbedder.new(model_id) do |cfg|
            cfg.epochs = @config[:epochs]
            cfg.batch_size = @config[:batch_size]
            cfg.learning_rate = @config[:learning_rate]
            cfg.callbacks << Callbacks::WebCallback.new(self)
          end

          @model.fit(train_file: @config[:train_file])
        end
      end
    end
  end
end
