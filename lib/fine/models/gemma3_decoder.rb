# frozen_string_literal: true

module Fine
  module Models
    # Gemma 3 decoder-only transformer
    #
    # Differences from Llama:
    # - Explicit head_dim config
    # - QK normalization (q_norm, k_norm)
    # - Pre/post feedforward layer norms
    # - Different GQA ratios
    class Gemma3Decoder < Base
      def initialize(config)
        super(config)

        @vocab_size = config.vocab_size
        @hidden_size = config.hidden_size
        @num_layers = config.num_hidden_layers
        @num_heads = config.num_attention_heads
        @num_kv_heads = config.num_key_value_heads || @num_heads
        @head_dim = config.head_dim
        @intermediate_size = config.intermediate_size
        @max_position_embeddings = config.max_position_embeddings || 32768
        @rms_norm_eps = config.rms_norm_eps || 1e-6
        @rope_theta = config.rope_theta || 1_000_000.0

        # Token embeddings
        @embed_tokens = Torch::NN::Embedding.new(@vocab_size, @hidden_size)

        # Transformer layers
        @layers = Torch::NN::ModuleList.new(
          @num_layers.times.map do
            Gemma3DecoderLayer.new(
              hidden_size: @hidden_size,
              num_heads: @num_heads,
              num_kv_heads: @num_kv_heads,
              head_dim: @head_dim,
              intermediate_size: @intermediate_size,
              rms_norm_eps: @rms_norm_eps,
              rope_theta: @rope_theta,
              max_position_embeddings: @max_position_embeddings
            )
          end
        )

        # Final layer norm
        @norm = RMSNorm.new(@hidden_size, eps: @rms_norm_eps)
      end

      def forward(input_ids, attention_mask: nil, position_ids: nil)
        batch_size, seq_length = input_ids.shape

        # Get token embeddings
        hidden_states = @embed_tokens.call(input_ids)

        # Normalize embeddings (Gemma specific)
        hidden_states = hidden_states * Math.sqrt(@hidden_size)

        # Create position IDs if not provided
        position_ids ||= Torch.arange(seq_length, device: input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create causal mask (must match dtype of hidden_states)
        causal_mask = create_causal_mask(seq_length, hidden_states.device, hidden_states.dtype)

        # Combine with attention mask if provided
        if attention_mask
          expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype)
          expanded_mask = expanded_mask.expand(-1, -1, seq_length, -1)
          causal_mask = causal_mask + (1.0 - expanded_mask) * -1e9
        end

        # Forward through layers
        @layers.each do |layer|
          hidden_states = layer.call(
            hidden_states,
            attention_mask: causal_mask,
            position_ids: position_ids
          )
        end

        # Final norm
        hidden_states = @norm.call(hidden_states)

        { last_hidden_state: hidden_states }
      end

      private

      def create_causal_mask(seq_length, device, dtype)
        mask = Torch.triu(
          Torch.ones(seq_length, seq_length, device: device, dtype: dtype) * -1e9,
          diagonal: 1
        )
        mask.unsqueeze(0).unsqueeze(0)
      end
    end

    # Single Gemma 3 decoder layer
    class Gemma3DecoderLayer < Torch::NN::Module
      def initialize(hidden_size:, num_heads:, num_kv_heads:, head_dim:,
                     intermediate_size:, rms_norm_eps:, rope_theta:, max_position_embeddings:)
        super()

        @self_attn = Gemma3Attention.new(
          hidden_size: hidden_size,
          num_heads: num_heads,
          num_kv_heads: num_kv_heads,
          head_dim: head_dim,
          rope_theta: rope_theta,
          max_position_embeddings: max_position_embeddings,
          rms_norm_eps: rms_norm_eps
        )

        @mlp = Gemma3MLP.new(
          hidden_size: hidden_size,
          intermediate_size: intermediate_size
        )

        @input_layernorm = RMSNorm.new(hidden_size, eps: rms_norm_eps)
        @post_attention_layernorm = RMSNorm.new(hidden_size, eps: rms_norm_eps)
        @pre_feedforward_layernorm = RMSNorm.new(hidden_size, eps: rms_norm_eps)
        @post_feedforward_layernorm = RMSNorm.new(hidden_size, eps: rms_norm_eps)
      end

      def forward(hidden_states, attention_mask: nil, position_ids: nil)
        # Self attention with residual
        residual = hidden_states
        hidden_states = @input_layernorm.call(hidden_states)
        hidden_states = @self_attn.call(
          hidden_states,
          attention_mask: attention_mask,
          position_ids: position_ids
        )
        hidden_states = @post_attention_layernorm.call(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = @pre_feedforward_layernorm.call(hidden_states)
        hidden_states = @mlp.call(hidden_states)
        hidden_states = @post_feedforward_layernorm.call(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states
      end
    end

    # Gemma 3 attention with QK normalization
    class Gemma3Attention < Torch::NN::Module
      def initialize(hidden_size:, num_heads:, num_kv_heads:, head_dim:,
                     rope_theta:, max_position_embeddings:, rms_norm_eps:)
        super()

        @num_heads = num_heads
        @num_kv_heads = num_kv_heads
        @head_dim = head_dim
        @num_key_value_groups = num_heads / num_kv_heads

        @q_proj = Torch::NN::Linear.new(hidden_size, num_heads * head_dim, bias: false)
        @k_proj = Torch::NN::Linear.new(hidden_size, num_kv_heads * head_dim, bias: false)
        @v_proj = Torch::NN::Linear.new(hidden_size, num_kv_heads * head_dim, bias: false)
        @o_proj = Torch::NN::Linear.new(num_heads * head_dim, hidden_size, bias: false)

        # QK normalization (Gemma 3 specific)
        @q_norm = RMSNorm.new(head_dim, eps: rms_norm_eps)
        @k_norm = RMSNorm.new(head_dim, eps: rms_norm_eps)

        @rotary_emb = RotaryEmbedding.new(head_dim, max_position_embeddings, rope_theta)
      end

      def forward(hidden_states, attention_mask: nil, position_ids: nil)
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = @q_proj.call(hidden_states)
        key_states = @k_proj.call(hidden_states)
        value_states = @v_proj.call(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, @num_heads, @head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, @num_kv_heads, @head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, @num_kv_heads, @head_dim).transpose(1, 2)

        # Apply QK normalization
        query_states = apply_qk_norm(query_states, @q_norm)
        key_states = apply_qk_norm(key_states, @k_norm)

        # Apply rotary embeddings
        cos, sin = @rotary_emb.call(value_states, position_ids)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        # Repeat KV heads for grouped-query attention
        if @num_key_value_groups > 1
          key_states = repeat_kv(key_states, @num_key_value_groups)
          value_states = repeat_kv(value_states, @num_key_value_groups)
        end

        # Attention with softcapping (Gemma uses sqrt(head_dim) scaling)
        scale = @head_dim ** -0.5
        attn_weights = Torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Apply causal mask
        attn_weights = attn_weights + attention_mask if attention_mask

        attn_weights = Torch::NN::Functional.softmax(attn_weights, dim: -1)
        attn_output = Torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous
        attn_output = attn_output.reshape(batch_size, seq_length, -1)

        @o_proj.call(attn_output)
      end

      private

      def apply_qk_norm(states, norm)
        # states: (batch, heads, seq, head_dim)
        # Need to apply norm per head
        batch, heads, seq, head_dim = states.shape
        states = states.transpose(1, 2).reshape(batch * seq, heads, head_dim)
        states = norm.call(states)
        states.reshape(batch, seq, heads, head_dim).transpose(1, 2)
      end

      def apply_rotary_pos_emb(x, cos, sin)
        x1 = x[0.., 0.., 0.., 0...(@head_dim / 2)]
        x2 = x[0.., 0.., 0.., (@head_dim / 2)..]
        rotated = Torch.cat([-x2, x1], dim: -1)
        (x * cos) + (rotated * sin)
      end

      def repeat_kv(x, n_rep)
        batch, num_kv_heads, seq_len, head_dim = x.shape
        return x if n_rep == 1

        x = x.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
      end
    end

    # Gemma 3 MLP with GELU activation (not SiLU like Llama)
    class Gemma3MLP < Torch::NN::Module
      def initialize(hidden_size:, intermediate_size:)
        super()

        @gate_proj = Torch::NN::Linear.new(hidden_size, intermediate_size, bias: false)
        @up_proj = Torch::NN::Linear.new(hidden_size, intermediate_size, bias: false)
        @down_proj = Torch::NN::Linear.new(intermediate_size, hidden_size, bias: false)
      end

      def forward(x)
        # GeGLU: gelu(gate) * up
        # Using GELU with tanh approximation as per Gemma config
        gate = Torch::NN::Functional.gelu(@gate_proj.call(x), approximate: "tanh")
        up = @up_proj.call(x)
        @down_proj.call(gate * up)
      end
    end
  end
end
