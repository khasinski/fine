# Installation

## Requirements

- Ruby 3.1+
- LibTorch 2.x
- libvips (for image processing)

## Quick Install

```bash
# Install system dependencies (macOS)
brew install pytorch libvips

# Install system dependencies (Ubuntu/Debian)
# sudo apt-get install libvips-dev
# Download LibTorch from https://pytorch.org/get-started/locally/

# Add to your Gemfile
gem 'fine'

# Install
bundle install
```

## Detailed Setup

### 1. Install LibTorch

**macOS (Homebrew):**
```bash
brew install pytorch
```

**Linux:**
Download from [pytorch.org](https://pytorch.org/get-started/locally/) and set:
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

**Verify:**
```bash
# Should show libtorch path
ls $(brew --prefix pytorch)/lib/libtorch* 2>/dev/null || echo "Check LibTorch installation"
```

### 2. Install libvips

**macOS:**
```bash
brew install vips
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libvips-dev
```

**Verify:**
```bash
vips --version
```

### 3. Install the Gem

```ruby
# Gemfile
gem 'fine'
```

```bash
bundle install
```

### 4. Verify Installation

```ruby
require 'fine'

puts Fine::VERSION
puts "Torch available: #{defined?(Torch)}"
puts "Device: #{Fine.device}"
```

## Troubleshooting

### LibTorch Version Mismatch

Error: `Incompatible LibTorch version`

The torch-rb gem requires a specific LibTorch version. Check compatibility:
```bash
# Check your LibTorch version
python3 -c "import torch; print(torch.__version__)"

# Or check brew
brew info pytorch
```

Update LibTorch or pin torch-rb to a compatible version.

### Missing libvips

Error: `cannot load such file -- vips`

```bash
# macOS
brew install vips

# Ubuntu
sudo apt-get install libvips-dev
```

### Native Extension Build Failures

```bash
# Ensure you have build tools
xcode-select --install  # macOS

# Reinstall with verbose output
gem install torch-rb -- --verbose
```

## GPU Support

### CUDA (NVIDIA)

Ensure CUDA toolkit is installed and LibTorch was built with CUDA support.

```ruby
Fine.configure do |config|
  config.device = "cuda"  # or "cuda:0" for specific GPU
end
```

### MPS (Apple Silicon)

Automatic on M1/M2/M3 Macs with supported PyTorch:

```ruby
puts Fine.device  # Should show "mps" on Apple Silicon
```

### CPU Only

```ruby
Fine.configure do |config|
  config.device = "cpu"
end
```
