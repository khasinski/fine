# frozen_string_literal: true

require "sinatra/base"
require "json"

require_relative "server/session/session_store"
require_relative "server/session/training_session"
require_relative "server/callbacks/web_callback"
require_relative "server/app"

module Fine
  module Server
    class << self
      def start(options = {})
        App.run!(
          host: options[:host] || "127.0.0.1",
          port: options[:port] || 9292,
          server: "puma"
        )
      end
    end
  end
end
