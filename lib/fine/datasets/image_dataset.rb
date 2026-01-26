# frozen_string_literal: true

module Fine
  module Datasets
    # Dataset for loading images from a directory structure
    #
    # Expected structure:
    #   data/
    #     class1/
    #       image1.jpg
    #       image2.jpg
    #     class2/
    #       image3.jpg
    #
    class ImageDataset
      include Enumerable

      attr_reader :images, :labels, :label_map, :inverse_label_map, :transforms

      IMAGE_EXTENSIONS = %w[.jpg .jpeg .png .webp .bmp .gif].freeze

      # Create dataset from a directory with class subdirectories
      #
      # @param path [String] Path to the root directory
      # @param transforms [Transforms::Compose, nil] Optional transforms to apply
      # @param validate [Boolean] Whether to validate directory structure
      # @return [ImageDataset]
      #
      # @example Expected directory structure
      #   # data/
      #   #   cats/
      #   #     cat1.jpg
      #   #     cat2.jpg
      #   #   dogs/
      #   #     dog1.jpg
      #   #     dog2.jpg
      #   dataset = ImageDataset.from_directory("data/", transforms: transforms)
      #
      def self.from_directory(path, transforms: nil, validate: true)
        Validators.validate_image_directory!(path) if validate

        images = []
        labels = []

        # Get sorted list of class directories
        label_names = Dir.children(path)
          .select { |f| File.directory?(File.join(path, f)) }
          .reject { |f| f.start_with?(".") }
          .sort

        raise DatasetError, "No class directories found in #{path}" if label_names.empty?

        # Build label map
        label_map = label_names.each_with_index.to_h

        # Collect images from each class directory
        label_names.each do |label_name|
          class_dir = File.join(path, label_name)
          label_id = label_map[label_name]

          Dir.glob(File.join(class_dir, "*")).each do |image_path|
            next unless image_file?(image_path)

            images << image_path
            labels << label_id
          end
        end

        raise DatasetError, "No images found in #{path}" if images.empty?

        new(images: images, labels: labels, label_map: label_map, transforms: transforms)
      end

      # Create dataset from explicit arrays
      #
      # @param images [Array<String>] Array of image paths
      # @param labels [Array<Integer, String>] Array of labels
      # @param label_map [Hash, nil] Optional mapping of label names to IDs
      # @param transforms [Transforms::Compose, nil] Optional transforms
      def initialize(images:, labels:, label_map: nil, transforms: nil)
        raise ArgumentError, "images and labels must have same length" if images.size != labels.size

        @images = images
        @transforms = transforms || default_transforms

        # Build label map if not provided
        if label_map
          @label_map = label_map
        else
          unique_labels = labels.uniq.sort
          @label_map = unique_labels.each_with_index.to_h
        end

        # Convert string labels to integers if needed
        @labels = labels.map do |label|
          label.is_a?(Integer) ? label : @label_map[label]
        end

        # Build inverse mapping
        @inverse_label_map = @label_map.invert
      end

      # Get a single item from the dataset
      #
      # @param index [Integer] Index of the item
      # @return [Hash] Hash with :pixel_values and :label keys
      def [](index)
        image = load_image(@images[index])
        image = @transforms.call(image)

        { pixel_values: image, label: @labels[index] }
      end

      # Number of items in the dataset
      def size
        @images.size
      end
      alias length size

      # Iterate over all items
      def each
        return enum_for(:each) unless block_given?

        size.times { |i| yield self[i] }
      end

      # Number of classes
      def num_classes
        @label_map.size
      end

      # Get class names in order
      def class_names
        @inverse_label_map.sort.map(&:last)
      end

      # Split dataset into train and validation sets
      #
      # @param test_size [Float] Fraction of data to use for validation (0.0-1.0)
      # @param shuffle [Boolean] Whether to shuffle before splitting
      # @param stratify [Boolean] Whether to maintain class distribution
      # @param seed [Integer, nil] Random seed for reproducibility
      # @return [Array<ImageDataset, ImageDataset>] Train and validation datasets
      def split(test_size: 0.2, shuffle: true, stratify: true, seed: nil)
        rng = seed ? Random.new(seed) : Random.new

        indices = (0...size).to_a
        indices = indices.shuffle(random: rng) if shuffle && !stratify

        if stratify
          train_indices, val_indices = stratified_split(indices, test_size, rng)
        else
          split_idx = (size * (1 - test_size)).round
          train_indices = indices[0...split_idx]
          val_indices = indices[split_idx..]
        end

        train_set = subset(train_indices)
        val_set = subset(val_indices)

        [train_set, val_set]
      end

      private

      def self.image_file?(path)
        return false unless File.file?(path)

        ext = File.extname(path).downcase
        IMAGE_EXTENSIONS.include?(ext)
      end

      def load_image(path)
        Vips::Image.new_from_file(path, access: :sequential)
      rescue Vips::Error => e
        raise ImageProcessingError.new(path, "Failed to load image: #{e.message}")
      end

      def default_transforms
        Transforms::Compose.new([
          Transforms::Resize.new(224),
          Transforms::ToTensor.new,
          Transforms::Normalize.new
        ])
      end

      def subset(indices)
        ImageDataset.new(
          images: indices.map { |i| @images[i] },
          labels: indices.map { |i| @labels[i] },
          label_map: @label_map,
          transforms: @transforms
        )
      end

      def stratified_split(indices, test_size, rng)
        train_indices = []
        val_indices = []

        # Group indices by label
        by_label = indices.group_by { |i| @labels[i] }

        by_label.each_value do |label_indices|
          shuffled = label_indices.shuffle(random: rng)
          split_idx = (shuffled.size * (1 - test_size)).round

          train_indices.concat(shuffled[0...split_idx])
          val_indices.concat(shuffled[split_idx..])
        end

        [train_indices.shuffle(random: rng), val_indices.shuffle(random: rng)]
      end
    end
  end
end
