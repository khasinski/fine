# frozen_string_literal: true

require "singleton"

module Fine
  module Server
    module Session
      # In-memory storage for training sessions
      class SessionStore
        include Singleton

        def initialize
          @sessions = {}
          @mutex = Mutex.new
        end

        def create(type:, config:)
          session = TrainingSession.new(type: type, config: config)
          @mutex.synchronize { @sessions[session.id] = session }
          session
        end

        def find(id)
          @mutex.synchronize { @sessions[id] }
        end

        def all
          @mutex.synchronize { @sessions.values }
        end

        def running
          all.select { |s| s.status == :running }
        end

        def completed
          all.select { |s| s.status == :completed }
        end

        def delete(id)
          @mutex.synchronize { @sessions.delete(id) }
        end

        def clear
          @mutex.synchronize { @sessions.clear }
        end
      end
    end
  end
end
