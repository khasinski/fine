# frozen_string_literal: true

module Fine
  module Models
    # SigLIP2 Vision Transformer Encoder
    #
    # Implements the vision encoder portion of SigLIP2 for image feature extraction.
    # Architecture follows the standard ViT with patch embedding, transformer blocks,
    # and pooling.
    class SigLIP2VisionEncoder < Base
      def initialize(config)
        super(config)

        @hidden_size = config.hidden_size
        @num_layers = config.num_hidden_layers
        @num_heads = config.num_attention_heads
        @intermediate_size = config.intermediate_size
        @image_size = config.image_size
        @patch_size = config.patch_size
        @num_channels = config.num_channels
        @layer_norm_eps = config.layer_norm_eps

        # Patch embedding
        @patch_embed = PatchEmbedding.new(
          image_size: @image_size,
          patch_size: @patch_size,
          in_channels: @num_channels,
          embed_dim: @hidden_size
        )

        # Position embedding (learnable)
        num_patches = (@image_size / @patch_size) ** 2
        @pos_embed = Torch::NN::Parameter.new(
          Torch.zeros(1, num_patches, @hidden_size)
        )

        # Transformer blocks
        @blocks = Torch::NN::ModuleList.new(
          @num_layers.times.map do
            TransformerBlock.new(
              hidden_size: @hidden_size,
              num_heads: @num_heads,
              intermediate_size: @intermediate_size,
              layer_norm_eps: @layer_norm_eps
            )
          end
        )

        # Final layer norm
        @norm = Torch::NN::LayerNorm.new(@hidden_size, eps: @layer_norm_eps)

        # Initialize position embedding
        init_pos_embed
      end

      def forward(pixel_values)
        # pixel_values: (batch, channels, height, width)

        # Patch embedding: (batch, num_patches, hidden_size)
        x = @patch_embed.call(pixel_values)

        # Add position embedding
        x = x + @pos_embed

        # Transformer blocks
        @blocks.each do |block|
          x = block.call(x)
        end

        # Final layer norm
        x = @norm.call(x)

        # Pool: take mean of all patch embeddings
        x.mean(dim: 1)
      end

      private

      def init_pos_embed
        # Initialize with normal distribution (truncated normal not available in torch.rb)
        # The values will be overwritten when loading pretrained weights
        Torch::NN::Init.normal!(@pos_embed, mean: 0.0, std: 0.02)
      end
    end

    # Patch embedding layer
    class PatchEmbedding < Torch::NN::Module
      def initialize(image_size:, patch_size:, in_channels:, embed_dim:)
        super()

        @image_size = image_size
        @patch_size = patch_size
        @num_patches = (image_size / patch_size) ** 2

        # Use conv2d for efficient patch extraction
        @proj = Torch::NN::Conv2d.new(
          in_channels, embed_dim, patch_size,
          stride: patch_size
        )
      end

      def forward(x)
        # x: (batch, channels, height, width)
        # output: (batch, num_patches, embed_dim)

        x = @proj.call(x) # (batch, embed_dim, h/patch, w/patch)
        x = x.flatten(2)   # (batch, embed_dim, num_patches)
        x.transpose(1, 2)  # (batch, num_patches, embed_dim)
      end
    end

    # Transformer block with self-attention and MLP
    class TransformerBlock < Torch::NN::Module
      def initialize(hidden_size:, num_heads:, intermediate_size:, layer_norm_eps:)
        super()

        @norm1 = Torch::NN::LayerNorm.new(hidden_size, eps: layer_norm_eps)
        @attn = Attention.new(hidden_size: hidden_size, num_heads: num_heads)
        @norm2 = Torch::NN::LayerNorm.new(hidden_size, eps: layer_norm_eps)
        @mlp = MLP.new(hidden_size: hidden_size, intermediate_size: intermediate_size)
      end

      def forward(x)
        # Pre-norm architecture
        x = x + @attn.call(@norm1.call(x))
        x = x + @mlp.call(@norm2.call(x))
        x
      end
    end

    # Multi-head self-attention
    class Attention < Torch::NN::Module
      def initialize(hidden_size:, num_heads:)
        super()

        @num_heads = num_heads
        @head_dim = hidden_size / num_heads
        @scale = @head_dim ** -0.5
        @hidden_size = hidden_size

        # Separate Q, K, V projections (matches HuggingFace)
        @q_proj = Torch::NN::Linear.new(hidden_size, hidden_size)
        @k_proj = Torch::NN::Linear.new(hidden_size, hidden_size)
        @v_proj = Torch::NN::Linear.new(hidden_size, hidden_size)
        @out_proj = Torch::NN::Linear.new(hidden_size, hidden_size)
      end

      def forward(x)
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V separately
        q = @q_proj.call(x)
        k = @k_proj.call(x)
        v = @v_proj.call(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = Torch.matmul(q, k.transpose(-2, -1)) * @scale
        attn = attn.softmax(dim: -1)

        # Apply attention to values
        out = Torch.matmul(attn, v) # (batch, heads, seq, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, @hidden_size)

        @out_proj.call(out)
      end
    end

    # MLP (feed-forward network)
    class MLP < Torch::NN::Module
      def initialize(hidden_size:, intermediate_size:)
        super()

        @fc1 = Torch::NN::Linear.new(hidden_size, intermediate_size)
        @act = Torch::NN::GELU.new
        @fc2 = Torch::NN::Linear.new(intermediate_size, hidden_size)
      end

      def forward(x)
        x = @fc1.call(x)
        x = @act.call(x)
        @fc2.call(x)
      end
    end
  end
end
