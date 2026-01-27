# frozen_string_literal: true

require "sinatra/base"
require "json"
require "securerandom"

module Fine
  module Server
    class App < Sinatra::Base
      set :root, File.dirname(__FILE__)
      set :public_folder, File.join(root, "public")
      set :views, File.join(root, "views")
      set :sessions, true
      set :session_secret, ENV.fetch("SESSION_SECRET") { SecureRandom.hex(32) }

      # Enable streaming for SSE
      set :server, :puma

      helpers do
        def json_response(data)
          content_type :json
          data.to_json
        end

        def session_store
          Session::SessionStore.instance
        end

        def h(text)
          Rack::Utils.escape_html(text.to_s)
        end
      end

      # Dashboard
      get "/" do
        @active_sessions = session_store.running
        @recent_sessions = session_store.all.reject { |s| s.status == :running }.last(10)
        erb :dashboard
      end

      # Training forms
      get "/training/new/:type" do
        @type = params[:type].to_sym
        unless %i[llm text_classifier image_classifier embedder].include?(@type)
          halt 404, "Unknown model type"
        end
        erb :"training/#{@type}"
      end

      # Start training
      post "/training/start" do
        type = params[:type]&.to_sym
        halt 400, json_response(error: "Missing type") unless type

        config = parse_training_config(params)
        session = session_store.create(type: type, config: config)
        session.start

        if request.accept?("text/html")
          redirect "/training/#{session.id}"
        else
          json_response(id: session.id, status: session.status)
        end
      end

      # Training progress page
      get "/training/:id" do
        @session = session_store.find(params[:id])
        halt 404, "Session not found" unless @session
        erb :"training/progress"
      end

      # Cancel training
      post "/training/:id/cancel" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session

        session.cancel
        json_response(status: session.status)
      end

      # SSE endpoint for live updates
      get "/events/:id", provides: "text/event-stream" do
        session = session_store.find(params[:id])
        halt 404 unless session

        content_type "text/event-stream"
        cache_control :no_cache

        stream(:keep_open) do |out|
          callback = ->(event, data) do
            out << "event: #{event}\n"
            out << "data: #{data.to_json}\n\n"
          end

          session.subscribe(&callback)

          # Send current state
          out << "event: state\n"
          out << "data: #{session.to_json.to_json}\n\n"

          # Keep connection open until training ends or client disconnects
          loop do
            break if session.status == :completed || session.status == :failed || session.status == :cancelled
            sleep 1
          end

          session.unsubscribe(callback)
        end
      end

      # Inference - Chat (LLM)
      get "/inference/chat/:id" do
        @session = session_store.find(params[:id])
        halt 404, "Session not found" unless @session
        halt 400, "Not an LLM session" unless @session.type == :llm
        erb :"inference/chat"
      end

      post "/inference/chat/:id" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session

        message = params[:message] || JSON.parse(request.body.read)["message"]
        response = session.chat(message)
        json_response(response: response)
      end

      # Inference - Classify (Text/Image)
      get "/inference/classify/:id" do
        @session = session_store.find(params[:id])
        halt 404, "Session not found" unless @session
        erb :"inference/classify"
      end

      post "/inference/classify/:id" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session

        input = params[:input] || params[:text]
        predictions = session.classify(input)
        json_response(predictions: predictions)
      end

      # Inference - Similarity (Embeddings)
      get "/inference/similarity/:id" do
        @session = session_store.find(params[:id])
        halt 404, "Session not found" unless @session
        halt 400, "Not an embedder session" unless @session.type == :embedder
        erb :"inference/similarity"
      end

      post "/inference/similarity/:id" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session

        body = JSON.parse(request.body.read)
        results = session.similarity_search(body["query"], body["corpus"], top_k: body["top_k"] || 5)
        json_response(results: results)
      end

      # Export
      get "/export/:id" do
        @session = session_store.find(params[:id])
        halt 404, "Session not found" unless @session
        erb :"export/index"
      end

      post "/export/:id/onnx" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session

        filename = session.export_onnx
        json_response(filename: filename, download_url: "/export/download/#{File.basename(filename)}")
      end

      post "/export/:id/gguf" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session
        halt 400, json_response(error: "GGUF export only for LLMs") unless session.type == :llm

        quantization = params[:quantization]&.to_sym || :f16
        filename = session.export_gguf(quantization: quantization)
        json_response(filename: filename, download_url: "/export/download/#{File.basename(filename)}")
      end

      get "/export/download/:filename" do
        exports_dir = File.join(Dir.pwd, "exports")
        filepath = File.join(exports_dir, params[:filename])

        halt 404, "File not found" unless File.exist?(filepath)
        send_file filepath, disposition: "attachment"
      end

      # Save model
      post "/training/:id/save" do
        session = session_store.find(params[:id])
        halt 404, json_response(error: "Session not found") unless session

        name = params[:name] || "model_#{session.id[0..7]}"
        path = session.save_model(name)
        json_response(path: path)
      end

      private

      def parse_training_config(params)
        {
          model_id: params[:model_id]&.to_s,
          train_file: extract_file_path(params[:train_file]),
          val_file: extract_file_path(params[:val_file]),
          train_dir: params[:train_dir]&.to_s,
          val_dir: params[:val_dir]&.to_s,
          epochs: (params[:epochs] || 3).to_i,
          batch_size: (params[:batch_size] || 4).to_i,
          learning_rate: (params[:learning_rate] || 2e-5).to_f,
          max_length: (params[:max_length] || 512).to_i,
          use_lora: params[:use_lora] == "true" || params[:use_lora] == "1" || params[:use_lora] == true,
          lora_rank: (params[:lora_rank] || 8).to_i,
          lora_alpha: (params[:lora_alpha] || 16).to_i
        }
      end

      def extract_file_path(param)
        return nil if param.nil?
        return param.to_s if param.is_a?(String)

        # Handle file upload hash (Sinatra/Rack style)
        if param.is_a?(Hash)
          if param[:tempfile]
            # Save uploaded file and return path
            uploads_dir = File.join(Dir.pwd, "uploads")
            FileUtils.mkdir_p(uploads_dir)
            filename = param[:filename] || "upload_#{Time.now.to_i}.jsonl"
            path = File.join(uploads_dir, filename)
            FileUtils.cp(param[:tempfile].path, path)
            return path
          elsif param["tempfile"]
            uploads_dir = File.join(Dir.pwd, "uploads")
            FileUtils.mkdir_p(uploads_dir)
            filename = param["filename"] || "upload_#{Time.now.to_i}.jsonl"
            path = File.join(uploads_dir, filename)
            FileUtils.cp(param["tempfile"].path, path)
            return path
          end
        end

        param.to_s
      end
    end
  end
end
