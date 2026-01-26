# frozen_string_literal: true

module Fine
  module Models
    # BERT/DistilBERT/DeBERTa Encoder
    #
    # Implements transformer encoder for text understanding tasks.
    # Supports loading pretrained weights from HuggingFace Hub.
    class BertEncoder < Base
      attr_reader :embeddings, :encoder, :pooler

      def initialize(config)
        super(config)

        @hidden_size = config.hidden_size
        @num_layers = config.num_hidden_layers
        @num_heads = config.num_attention_heads
        @intermediate_size = config.intermediate_size
        @vocab_size = config.vocab_size
        @max_position_embeddings = config.max_position_embeddings
        @type_vocab_size = config.type_vocab_size || 2
        @layer_norm_eps = config.layer_norm_eps
        @hidden_dropout_prob = config.hidden_dropout_prob || 0.1

        # Embeddings
        @word_embeddings = Torch::NN::Embedding.new(@vocab_size, @hidden_size)
        @position_embeddings = Torch::NN::Embedding.new(@max_position_embeddings, @hidden_size)
        @token_type_embeddings = Torch::NN::Embedding.new(@type_vocab_size, @hidden_size)
        @embeddings_layer_norm = Torch::NN::LayerNorm.new(@hidden_size, eps: @layer_norm_eps)
        @embeddings_dropout = Torch::NN::Dropout.new(p: @hidden_dropout_prob)

        # Transformer layers
        @layers = Torch::NN::ModuleList.new(
          @num_layers.times.map do
            BertLayer.new(
              hidden_size: @hidden_size,
              num_heads: @num_heads,
              intermediate_size: @intermediate_size,
              layer_norm_eps: @layer_norm_eps,
              hidden_dropout_prob: @hidden_dropout_prob
            )
          end
        )

        # Pooler (for [CLS] token representation)
        @pooler_dense = Torch::NN::Linear.new(@hidden_size, @hidden_size)
        @pooler_activation = Torch::NN::Tanh.new
      end

      def forward(input_ids, attention_mask: nil, token_type_ids: nil)
        batch_size, seq_length = input_ids.shape

        # Create position IDs
        position_ids = Torch.arange(seq_length, device: input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Default token type IDs to zeros
        token_type_ids ||= Torch.zeros_like(input_ids)

        # Embeddings
        word_embeds = @word_embeddings.call(input_ids)
        position_embeds = @position_embeddings.call(position_ids)
        token_type_embeds = @token_type_embeddings.call(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = @embeddings_layer_norm.call(embeddings)
        embeddings = @embeddings_dropout.call(embeddings)

        # Create attention mask for transformer
        # Convert from (batch, seq) to (batch, 1, 1, seq) for broadcasting
        if attention_mask
          extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
          extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else
          extended_attention_mask = nil
        end

        # Transformer layers
        hidden_states = embeddings
        @layers.each do |layer|
          hidden_states = layer.call(hidden_states, attention_mask: extended_attention_mask)
        end

        # Pool the [CLS] token (first token)
        cls_output = hidden_states[0.., 0, 0..]
        pooled_output = @pooler_dense.call(cls_output)
        pooled_output = @pooler_activation.call(pooled_output)

        {
          last_hidden_state: hidden_states,
          pooler_output: pooled_output
        }
      end

      # Get the [CLS] token embedding (useful for classification)
      def get_pooled_output(input_ids, attention_mask: nil, token_type_ids: nil)
        output = forward(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids)
        output[:pooler_output]
      end

      # Get mean of all token embeddings (useful for sentence embeddings)
      def get_mean_output(input_ids, attention_mask: nil, token_type_ids: nil)
        output = forward(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids)
        hidden_states = output[:last_hidden_state]

        if attention_mask
          # Mask padding tokens before taking mean
          mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float
          sum_embeddings = (hidden_states * mask).sum(dim: 1)
          sum_mask = mask.sum(dim: 1).clamp(min: 1e-9)
          sum_embeddings / sum_mask
        else
          hidden_states.mean(dim: 1)
        end
      end
    end

    # Single BERT transformer layer
    class BertLayer < Torch::NN::Module
      def initialize(hidden_size:, num_heads:, intermediate_size:, layer_norm_eps:, hidden_dropout_prob:)
        super()

        @attention = BertAttention.new(
          hidden_size: hidden_size,
          num_heads: num_heads,
          dropout: hidden_dropout_prob
        )
        @attention_layer_norm = Torch::NN::LayerNorm.new(hidden_size, eps: layer_norm_eps)
        @attention_dropout = Torch::NN::Dropout.new(p: hidden_dropout_prob)

        @intermediate = Torch::NN::Linear.new(hidden_size, intermediate_size)
        @intermediate_act = Torch::NN::GELU.new
        @output = Torch::NN::Linear.new(intermediate_size, hidden_size)
        @output_layer_norm = Torch::NN::LayerNorm.new(hidden_size, eps: layer_norm_eps)
        @output_dropout = Torch::NN::Dropout.new(p: hidden_dropout_prob)
      end

      def forward(hidden_states, attention_mask: nil)
        # Self-attention with residual
        attention_output = @attention.call(hidden_states, attention_mask: attention_mask)
        attention_output = @attention_dropout.call(attention_output)
        hidden_states = @attention_layer_norm.call(hidden_states + attention_output)

        # FFN with residual
        intermediate_output = @intermediate.call(hidden_states)
        intermediate_output = @intermediate_act.call(intermediate_output)
        layer_output = @output.call(intermediate_output)
        layer_output = @output_dropout.call(layer_output)
        @output_layer_norm.call(hidden_states + layer_output)
      end
    end

    # BERT multi-head self-attention
    class BertAttention < Torch::NN::Module
      def initialize(hidden_size:, num_heads:, dropout:)
        super()

        @num_heads = num_heads
        @head_dim = hidden_size / num_heads
        @scale = @head_dim ** -0.5

        @query = Torch::NN::Linear.new(hidden_size, hidden_size)
        @key = Torch::NN::Linear.new(hidden_size, hidden_size)
        @value = Torch::NN::Linear.new(hidden_size, hidden_size)
        @out = Torch::NN::Linear.new(hidden_size, hidden_size)
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      def forward(hidden_states, attention_mask: nil)
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        q = @query.call(hidden_states)
        k = @key.call(hidden_states)
        v = @value.call(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, @num_heads, @head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, @num_heads, @head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, @num_heads, @head_dim).transpose(1, 2)

        # Attention scores
        scores = Torch.matmul(q, k.transpose(-2, -1)) * @scale

        # Apply attention mask
        scores = scores + attention_mask if attention_mask

        # Softmax and dropout
        attn_probs = Torch::NN::Functional.softmax(scores, dim: -1)
        attn_probs = @dropout.call(attn_probs)

        # Apply attention to values
        context = Torch.matmul(attn_probs, v)

        # Reshape back
        context = context.transpose(1, 2).contiguous.view(batch_size, seq_length, -1)

        @out.call(context)
      end
    end
  end
end
