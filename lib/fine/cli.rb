# frozen_string_literal: true

module Fine
  # Command-line interface for Fine
  class CLI
    class << self
      def run(args)
        command = args.shift

        case command
        when "server"
          start_server(args)
        when "version", "-v", "--version"
          puts "Fine #{Fine::VERSION}"
        when "help", "-h", "--help", nil
          print_help
        else
          puts "Unknown command: #{command}"
          print_help
          exit 1
        end
      end

      private

      def start_server(args)
        require "fine/server"

        options = parse_server_options(args)

        puts "Starting Fine server on http://#{options[:host]}:#{options[:port]}"
        puts "Press Ctrl+C to stop"
        puts

        Fine::Server.start(options)
      end

      def parse_server_options(args)
        options = {
          host: "127.0.0.1",
          port: 9292
        }

        i = 0
        while i < args.length
          case args[i]
          when "-p", "--port"
            options[:port] = args[i + 1].to_i
            i += 2
          when "-h", "--host"
            options[:host] = args[i + 1]
            i += 2
          when "-b", "--bind"
            options[:host] = args[i + 1]
            i += 2
          else
            i += 1
          end
        end

        options
      end

      def print_help
        puts <<~HELP
          Fine - Fine-tune ML models with Ruby

          Usage:
            fine <command> [options]

          Commands:
            server    Start the web UI for training and monitoring
            version   Show version
            help      Show this help

          Server options:
            -p, --port PORT    Port to listen on (default: 9292)
            -h, --host HOST    Host to bind to (default: 127.0.0.1)

          Examples:
            fine server                  # Start on localhost:9292
            fine server -p 3000          # Start on port 3000
            fine server -h 0.0.0.0       # Bind to all interfaces
        HELP
      end
    end
  end
end
