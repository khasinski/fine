# frozen_string_literal: true

module Fine
  module Hub
    # Downloads models from Hugging Face Hub
    class ModelDownloader
      HF_HUB_URL = "https://huggingface.co"
      REQUIRED_FILES = %w[config.json].freeze
      MODEL_FILES = %w[model.safetensors].freeze
      OPTIONAL_FILES = %w[preprocessor_config.json tokenizer_config.json tokenizer.json].freeze

      attr_reader :model_id, :cache_path

      def initialize(model_id, cache_dir: nil)
        @model_id = model_id
        @cache_dir = cache_dir || Fine.cache_dir
        @cache_path = File.join(@cache_dir, "models", model_id.tr("/", "--"))
      end

      # Download model files and return the cache path
      def download(force: false)
        return @cache_path if cached? && !force

        FileUtils.mkdir_p(@cache_path)

        download_required_files
        download_optional_files

        @cache_path
      end

      # Check if model is already cached
      def cached?
        return false unless File.directory?(@cache_path)

        # Check required files
        return false unless REQUIRED_FILES.all? { |f| File.exist?(File.join(@cache_path, f)) }

        # Check for model weights (single file or sharded)
        has_single_weights = File.exist?(File.join(@cache_path, "model.safetensors"))
        has_sharded_weights = Dir.glob(File.join(@cache_path, "model-*.safetensors")).any?

        has_single_weights || has_sharded_weights
      end

      # Get path to a specific file
      def file_path(filename)
        File.join(@cache_path, filename)
      end

      private

      def download_required_files
        REQUIRED_FILES.each do |filename|
          download_file(filename)
        end

        # Download model weights (try single file first, then sharded)
        download_model_weights
      end

      def download_model_weights
        # Try single model file first
        begin
          download_file("model.safetensors")
          return
        rescue ModelNotFoundError
          # Single file not found, try sharded
        end

        # Download index file to find shards
        download_file("model.safetensors.index.json", required: false)
        index_path = File.join(@cache_path, "model.safetensors.index.json")

        if File.exist?(index_path)
          index = JSON.parse(File.read(index_path))
          weight_files = index["weight_map"].values.uniq

          weight_files.each do |filename|
            download_file(filename)
          end
        else
          raise ModelNotFoundError.new(@model_id, "No model weights found")
        end
      end

      def download_optional_files
        OPTIONAL_FILES.each do |filename|
          download_file(filename, required: false)
        end
      end

      def download_file(filename, required: true)
        local_path = File.join(@cache_path, filename)
        return if File.exist?(local_path)

        url = file_url(filename)

        begin
          puts "Downloading #{filename}..." if Fine.configuration&.progress_bar != false

          headers = { "User-Agent" => "fine-ruby/#{Fine::VERSION}" }

          # Add HuggingFace token if available
          if (token = hf_token)
            headers["Authorization"] = "Bearer #{token}"
          end

          tempfile = Down.download(url, headers: headers)

          FileUtils.mv(tempfile.path, local_path)
        rescue Down::NotFound
          raise ModelNotFoundError.new(@model_id, "File not found: #{filename}") if required
        rescue Down::Error => e
          raise ModelNotFoundError.new(@model_id, "Download failed: #{e.message}") if required
        end
      end

      def hf_token
        # Check environment variable first
        return ENV["HF_TOKEN"] if ENV["HF_TOKEN"]
        return ENV["HUGGING_FACE_HUB_TOKEN"] if ENV["HUGGING_FACE_HUB_TOKEN"]

        # Check standard HuggingFace cache location
        token_path = File.expand_path("~/.cache/huggingface/token")
        return File.read(token_path).strip if File.exist?(token_path)

        nil
      end

      def file_url(filename)
        "#{HF_HUB_URL}/#{@model_id}/resolve/main/#{filename}"
      end
    end
  end
end
