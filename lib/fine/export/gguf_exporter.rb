# frozen_string_literal: true

module Fine
  module Export
    # Export LLMs to GGUF format for llama.cpp, ollama, etc.
    #
    # @example Basic export
    #   llm = Fine::LLM.load("my_llama")
    #   Fine::Export::GGUFExporter.export(llm, "model.gguf")
    #
    # @example With quantization
    #   Fine::Export::GGUFExporter.export(
    #     llm,
    #     "model-q4.gguf",
    #     quantization: :q4_0
    #   )
    class GGUFExporter
      # GGUF magic number and version
      GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
      GGUF_VERSION = 3

      # GGUF value types
      GGUF_TYPE_UINT8 = 0
      GGUF_TYPE_INT8 = 1
      GGUF_TYPE_UINT16 = 2
      GGUF_TYPE_INT16 = 3
      GGUF_TYPE_UINT32 = 4
      GGUF_TYPE_INT32 = 5
      GGUF_TYPE_FLOAT32 = 6
      GGUF_TYPE_BOOL = 7
      GGUF_TYPE_STRING = 8
      GGUF_TYPE_ARRAY = 9
      GGUF_TYPE_UINT64 = 10
      GGUF_TYPE_INT64 = 11
      GGUF_TYPE_FLOAT64 = 12

      # GGML tensor types
      GGML_TYPE_F32 = 0
      GGML_TYPE_F16 = 1
      GGML_TYPE_Q4_0 = 2
      GGML_TYPE_Q4_1 = 3
      GGML_TYPE_Q5_0 = 6
      GGML_TYPE_Q5_1 = 7
      GGML_TYPE_Q8_0 = 8
      GGML_TYPE_Q8_1 = 9
      GGML_TYPE_Q2_K = 10
      GGML_TYPE_Q3_K = 11
      GGML_TYPE_Q4_K = 12
      GGML_TYPE_Q5_K = 13
      GGML_TYPE_Q6_K = 14
      GGML_TYPE_Q8_K = 15

      QUANTIZATION_TYPES = {
        f32: GGML_TYPE_F32,
        f16: GGML_TYPE_F16,
        q4_0: GGML_TYPE_Q4_0,
        q4_1: GGML_TYPE_Q4_1,
        q5_0: GGML_TYPE_Q5_0,
        q5_1: GGML_TYPE_Q5_1,
        q8_0: GGML_TYPE_Q8_0,
        q4_k: GGML_TYPE_Q4_K,
        q5_k: GGML_TYPE_Q5_K,
        q6_k: GGML_TYPE_Q6_K
      }.freeze

      class << self
        # Export a Fine::LLM to GGUF format
        #
        # @param llm [Fine::LLM] The LLM to export
        # @param output_path [String] Path for the output GGUF file
        # @param quantization [Symbol] Quantization type (:f16, :q4_0, :q4_k, :q8_0, etc.)
        # @param metadata [Hash] Additional metadata to include
        def export(llm, output_path, quantization: :f16, metadata: {})
          unless llm.is_a?(Fine::LLM)
            raise ExportError, "GGUF export only supports Fine::LLM models"
          end

          unless llm.model
            raise ExportError, "Model not loaded or trained"
          end

          exporter = new(llm, output_path, quantization, metadata)
          exporter.export
        end
      end

      def initialize(llm, output_path, quantization, metadata)
        @llm = llm
        @output_path = output_path
        @quantization = quantization
        @metadata = metadata
        @model = llm.model
        @config = extract_config
      end

      def export
        File.open(@output_path, "wb") do |file|
          @file = file

          write_header
          write_metadata
          write_tensors
        end

        @output_path
      end

      private

      def extract_config
        model_config = @model.config

        {
          vocab_size: model_config.vocab_size,
          hidden_size: model_config.hidden_size,
          intermediate_size: model_config.intermediate_size,
          num_hidden_layers: model_config.num_hidden_layers,
          num_attention_heads: model_config.num_attention_heads,
          num_key_value_heads: model_config.num_key_value_heads || model_config.num_attention_heads,
          max_position_embeddings: model_config.max_position_embeddings || 2048,
          rms_norm_eps: model_config.rms_norm_eps || 1e-6,
          rope_theta: model_config.rope_theta || 10000.0
        }
      end

      def write_header
        # Magic number
        @file.write([GGUF_MAGIC].pack("V"))

        # Version
        @file.write([GGUF_VERSION].pack("V"))

        # Tensor count (will be updated later)
        @tensor_count_pos = @file.pos
        @file.write([0].pack("Q<"))

        # Metadata KV count (will be updated later)
        @kv_count_pos = @file.pos
        @file.write([0].pack("Q<"))
      end

      def write_metadata
        kv_count = 0

        # Architecture
        write_string_kv("general.architecture", "llama")
        kv_count += 1

        write_string_kv("general.name", @llm.model_id || "fine-tuned-model")
        kv_count += 1

        # Model parameters
        write_uint32_kv("llama.context_length", @config[:max_position_embeddings])
        kv_count += 1

        write_uint32_kv("llama.embedding_length", @config[:hidden_size])
        kv_count += 1

        write_uint32_kv("llama.block_count", @config[:num_hidden_layers])
        kv_count += 1

        write_uint32_kv("llama.feed_forward_length", @config[:intermediate_size])
        kv_count += 1

        write_uint32_kv("llama.attention.head_count", @config[:num_attention_heads])
        kv_count += 1

        write_uint32_kv("llama.attention.head_count_kv", @config[:num_key_value_heads])
        kv_count += 1

        write_float32_kv("llama.rope.freq_base", @config[:rope_theta])
        kv_count += 1

        write_float32_kv("llama.attention.layer_norm_rms_epsilon", @config[:rms_norm_eps])
        kv_count += 1

        # Tokenizer info (if available)
        if @llm.tokenizer
          write_string_kv("tokenizer.ggml.model", "llama")
          kv_count += 1

          if @llm.tokenizer.respond_to?(:vocab_size)
            write_uint32_kv("llama.vocab_size", @llm.tokenizer.vocab_size)
            kv_count += 1
          end
        end

        # Custom metadata
        @metadata.each do |key, value|
          case value
          when String
            write_string_kv("general.#{key}", value)
          when Integer
            write_uint32_kv("general.#{key}", value)
          when Float
            write_float32_kv("general.#{key}", value)
          end
          kv_count += 1
        end

        # Update KV count
        current_pos = @file.pos
        @file.seek(@kv_count_pos)
        @file.write([kv_count].pack("Q<"))
        @file.seek(current_pos)
      end

      def write_tensors
        tensor_count = 0
        tensor_infos = []
        tensor_data = []

        state_dict = @model.state_dict

        state_dict.each do |name, tensor|
          gguf_name = convert_tensor_name(name)
          next unless gguf_name

          # Quantize tensor
          quantized, dtype = quantize_tensor(tensor, name)

          tensor_infos << {
            name: gguf_name,
            dims: tensor.shape.reverse,  # GGUF uses reversed dimensions
            dtype: dtype
          }

          tensor_data << quantized
          tensor_count += 1
        end

        # Write tensor infos
        tensor_infos.each do |info|
          write_tensor_info(info)
        end

        # Alignment padding
        align_to(32)

        # Write tensor data
        tensor_data.each_with_index do |data, idx|
          align_to(32)
          @file.write(data)
        end

        # Update tensor count
        current_pos = @file.pos
        @file.seek(@tensor_count_pos)
        @file.write([tensor_count].pack("Q<"))
        @file.seek(current_pos)
      end

      def convert_tensor_name(torch_name)
        # Map torch.rb/HuggingFace names to GGUF names
        name = torch_name.dup

        mappings = {
          "decoder.embed_tokens.weight" => "token_embd.weight",
          "decoder.norm.weight" => "output_norm.weight",
          "lm_head.weight" => "output.weight"
        }

        return mappings[name] if mappings.key?(name)

        # Layer mappings
        if name =~ /decoder\.layers\.(\d+)\./
          layer_num = $1

          layer_mappings = {
            "input_layernorm.weight" => "blk.#{layer_num}.attn_norm.weight",
            "post_attention_layernorm.weight" => "blk.#{layer_num}.ffn_norm.weight",
            "self_attn.q_proj.weight" => "blk.#{layer_num}.attn_q.weight",
            "self_attn.k_proj.weight" => "blk.#{layer_num}.attn_k.weight",
            "self_attn.v_proj.weight" => "blk.#{layer_num}.attn_v.weight",
            "self_attn.o_proj.weight" => "blk.#{layer_num}.attn_output.weight",
            "mlp.gate_proj.weight" => "blk.#{layer_num}.ffn_gate.weight",
            "mlp.up_proj.weight" => "blk.#{layer_num}.ffn_up.weight",
            "mlp.down_proj.weight" => "blk.#{layer_num}.ffn_down.weight"
          }

          suffix = name.sub(/decoder\.layers\.\d+\./, "")
          return layer_mappings[suffix]
        end

        nil  # Skip unknown tensors
      end

      def quantize_tensor(tensor, name)
        tensor = tensor.cpu.contiguous

        # Always keep embeddings and norms in higher precision
        if name.include?("embed") || name.include?("norm") || name.include?("lm_head")
          return [tensor_to_f16(tensor), GGML_TYPE_F16]
        end

        case @quantization
        when :f32
          [tensor_to_f32(tensor), GGML_TYPE_F32]
        when :f16
          [tensor_to_f16(tensor), GGML_TYPE_F16]
        when :q8_0
          quantize_q8_0(tensor)
        when :q4_0
          quantize_q4_0(tensor)
        when :q4_k, :q5_k, :q6_k
          # K-quants are more complex, fall back to Q8 for now
          quantize_q8_0(tensor)
        else
          [tensor_to_f16(tensor), GGML_TYPE_F16]
        end
      end

      def tensor_to_f32(tensor)
        tensor.to(:float32).data_ptr_bytes
      end

      def tensor_to_f16(tensor)
        tensor.to(:float16).data_ptr_bytes
      end

      def quantize_q8_0(tensor)
        # Q8_0: 8-bit quantization with block size 32
        block_size = 32
        data = tensor.to(:float32).flatten.to_a

        quantized = []

        data.each_slice(block_size) do |block|
          block = block + [0.0] * (block_size - block.size) if block.size < block_size

          # Find scale (max absolute value)
          max_abs = block.map(&:abs).max
          scale = max_abs / 127.0
          scale = 1.0 if scale == 0

          # Quantize
          quantized << [scale].pack("e")  # float16 scale
          block.each do |val|
            q = (val / scale).round.clamp(-128, 127)
            quantized << [q].pack("c")
          end
        end

        [quantized.join, GGML_TYPE_Q8_0]
      end

      def quantize_q4_0(tensor)
        # Q4_0: 4-bit quantization with block size 32
        block_size = 32
        data = tensor.to(:float32).flatten.to_a

        quantized = []

        data.each_slice(block_size) do |block|
          block = block + [0.0] * (block_size - block.size) if block.size < block_size

          # Find scale
          max_abs = block.map(&:abs).max
          scale = max_abs / 7.0
          scale = 1.0 if scale == 0

          # Quantize to 4-bit
          quantized << [scale].pack("e")  # float16 scale

          block.each_slice(2) do |pair|
            q0 = ((pair[0] / scale).round.clamp(-8, 7) + 8) & 0x0F
            q1 = ((pair[1] / scale).round.clamp(-8, 7) + 8) & 0x0F
            quantized << [(q0 | (q1 << 4))].pack("C")
          end
        end

        [quantized.join, GGML_TYPE_Q4_0]
      end

      def write_tensor_info(info)
        # Name
        write_string(info[:name])

        # Number of dimensions
        @file.write([info[:dims].size].pack("V"))

        # Dimensions
        info[:dims].each do |dim|
          @file.write([dim].pack("Q<"))
        end

        # Type
        @file.write([info[:dtype]].pack("V"))

        # Offset (will be calculated later, write 0 for now)
        @file.write([0].pack("Q<"))
      end

      def write_string_kv(key, value)
        write_string(key)
        @file.write([GGUF_TYPE_STRING].pack("V"))
        write_string(value)
      end

      def write_uint32_kv(key, value)
        write_string(key)
        @file.write([GGUF_TYPE_UINT32].pack("V"))
        @file.write([value].pack("V"))
      end

      def write_float32_kv(key, value)
        write_string(key)
        @file.write([GGUF_TYPE_FLOAT32].pack("V"))
        @file.write([value].pack("e"))
      end

      def write_string(str)
        @file.write([str.bytesize].pack("Q<"))
        @file.write(str)
      end

      def align_to(alignment)
        current = @file.pos
        padding = (alignment - (current % alignment)) % alignment
        @file.write("\x00" * padding) if padding > 0
      end
    end
  end
end
