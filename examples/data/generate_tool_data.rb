#!/usr/bin/env ruby
# frozen_string_literal: true

# Generate diverse Ollama tool calling training data

require "json"

# Tool definitions with example usages
TOOLS = {
  get_weather: {
    description: "Get current weather for a location",
    params: { location: "string, required" },
    examples: [
      { q: "What's the weather in Tokyo?", args: { location: "Tokyo" } },
      { q: "Is it raining in Seattle?", args: { location: "Seattle" } },
      { q: "Check the forecast for Paris", args: { location: "Paris" } },
      { q: "How cold is it in Moscow?", args: { location: "Moscow" } },
      { q: "Weather conditions in Sydney", args: { location: "Sydney" } },
      { q: "Temperature in New York City", args: { location: "New York City" } },
      { q: "What's it like outside in London?", args: { location: "London" } },
      { q: "Is it sunny in Miami?", args: { location: "Miami" } },
      { q: "Current weather for Berlin", args: { location: "Berlin" } },
      { q: "How's the weather in San Francisco?", args: { location: "San Francisco" } },
      { q: "Will I need an umbrella in Vancouver?", args: { location: "Vancouver" } },
      { q: "Check if it's snowing in Denver", args: { location: "Denver" } },
      { q: "What's the temperature in Chicago?", args: { location: "Chicago" } },
      { q: "Weather report for Los Angeles", args: { location: "Los Angeles" } },
      { q: "Is it hot in Phoenix today?", args: { location: "Phoenix" } },
    ]
  },
  calculate: {
    description: "Evaluate a math expression",
    params: { expression: "string, required" },
    examples: [
      { q: "Calculate 25 * 4 + 10", args: { expression: "25 * 4 + 10" } },
      { q: "What's 15% of 200?", args: { expression: "0.15 * 200" } },
      { q: "Find the square root of 144", args: { expression: "sqrt(144)" } },
      { q: "2 to the power of 10", args: { expression: "2 ** 10" } },
      { q: "Divide 1000 by 8", args: { expression: "1000 / 8" } },
      { q: "What is 123 + 456?", args: { expression: "123 + 456" } },
      { q: "Calculate the area of a circle with radius 5", args: { expression: "3.14159 * 5 * 5" } },
      { q: "How much is 99 times 99?", args: { expression: "99 * 99" } },
      { q: "What's 50 divided by 7?", args: { expression: "50 / 7" } },
      { q: "Calculate 3 factorial", args: { expression: "3 * 2 * 1" } },
      { q: "What's 1000 minus 347?", args: { expression: "1000 - 347" } },
      { q: "Find 20% of 500", args: { expression: "0.20 * 500" } },
      { q: "Calculate 8 cubed", args: { expression: "8 ** 3" } },
      { q: "What is sqrt(256)?", args: { expression: "sqrt(256)" } },
      { q: "How much is 75 + 25 * 2?", args: { expression: "75 + 25 * 2" } },
    ]
  },
  search_web: {
    description: "Search the web for information",
    params: { query: "string, required" },
    examples: [
      { q: "Search for Ruby tutorials", args: { query: "Ruby tutorials" } },
      { q: "Find information about machine learning", args: { query: "machine learning" } },
      { q: "Look up the population of Japan", args: { query: "population of Japan" } },
      { q: "Search for Python documentation", args: { query: "Python documentation" } },
      { q: "Find recipes for pasta", args: { query: "pasta recipes" } },
      { q: "Search the latest news about AI", args: { query: "latest AI news" } },
      { q: "Look up reviews for iPhone 15", args: { query: "iPhone 15 reviews" } },
      { q: "Find hotels in Barcelona", args: { query: "hotels in Barcelona" } },
      { q: "Search for JavaScript frameworks", args: { query: "JavaScript frameworks" } },
      { q: "Look up symptoms of flu", args: { query: "flu symptoms" } },
      { q: "Find the best coffee shops nearby", args: { query: "best coffee shops" } },
      { q: "Search for remote job listings", args: { query: "remote jobs" } },
      { q: "Find hiking trails in Colorado", args: { query: "hiking trails Colorado" } },
      { q: "Look up electric car comparisons", args: { query: "electric car comparison" } },
      { q: "Search for meditation apps", args: { query: "meditation apps" } },
    ]
  },
  send_email: {
    description: "Send an email",
    params: { to: "string, required", subject: "string, required", body: "string, required" },
    examples: [
      { q: "Send email to john@example.com about the meeting tomorrow", args: { to: "john@example.com", subject: "Meeting Tomorrow", body: "Hi, I wanted to discuss the meeting scheduled for tomorrow." } },
      { q: "Email sarah@work.com regarding the project update", args: { to: "sarah@work.com", subject: "Project Update", body: "Hi Sarah, Here's the latest update on our project." } },
      { q: "Send a thank you note to boss@company.com", args: { to: "boss@company.com", subject: "Thank You", body: "Thank you for the opportunity and support." } },
      { q: "Email team@startup.io about the launch", args: { to: "team@startup.io", subject: "Launch Day", body: "Team, today is launch day! Let's make it great." } },
      { q: "Send meeting notes to alice@corp.com", args: { to: "alice@corp.com", subject: "Meeting Notes", body: "Here are the notes from today's meeting." } },
    ]
  },
  create_event: {
    description: "Create a calendar event",
    params: { title: "string, required", date: "string, required", time: "string, required" },
    examples: [
      { q: "Create event for Friday at 3pm called Team Standup", args: { title: "Team Standup", date: "Friday", time: "3pm" } },
      { q: "Schedule a dentist appointment for Monday at 10am", args: { title: "Dentist Appointment", date: "Monday", time: "10am" } },
      { q: "Add lunch meeting to calendar for tomorrow at noon", args: { title: "Lunch Meeting", date: "tomorrow", time: "12pm" } },
      { q: "Create a reminder for gym on Wednesday 6pm", args: { title: "Gym", date: "Wednesday", time: "6pm" } },
      { q: "Schedule interview for next Tuesday at 2pm", args: { title: "Interview", date: "next Tuesday", time: "2pm" } },
    ]
  },
  set_reminder: {
    description: "Set a reminder",
    params: { message: "string, required" },
    examples: [
      { q: "Remind me to buy milk", args: { message: "buy milk" } },
      { q: "Set a reminder to call mom", args: { message: "call mom" } },
      { q: "Remind me to submit the report", args: { message: "submit the report" } },
      { q: "Set reminder to water the plants", args: { message: "water the plants" } },
      { q: "Remind me to take my medication", args: { message: "take medication" } },
    ]
  },
  translate: {
    description: "Translate text to another language",
    params: { text: "string, required", target_language: "string, required" },
    examples: [
      { q: "Translate hello to Spanish", args: { text: "hello", target_language: "Spanish" } },
      { q: "How do you say goodbye in French?", args: { text: "goodbye", target_language: "French" } },
      { q: "Translate 'thank you' to Japanese", args: { text: "thank you", target_language: "Japanese" } },
      { q: "Say 'I love you' in Italian", args: { text: "I love you", target_language: "Italian" } },
      { q: "What's 'good morning' in German?", args: { text: "good morning", target_language: "German" } },
    ]
  },
  get_time: {
    description: "Get the current time in a timezone",
    params: { timezone: "string, required" },
    examples: [
      { q: "What time is it in Tokyo?", args: { timezone: "Asia/Tokyo" } },
      { q: "Current time in London", args: { timezone: "Europe/London" } },
      { q: "What's the time in New York?", args: { timezone: "America/New_York" } },
      { q: "Time in Sydney now", args: { timezone: "Australia/Sydney" } },
      { q: "What time is it in Paris?", args: { timezone: "Europe/Paris" } },
    ]
  },
  read_file: {
    description: "Read contents of a file",
    params: { path: "string, required" },
    examples: [
      { q: "Read the config.json file", args: { path: "config.json" } },
      { q: "Show me the contents of README.md", args: { path: "README.md" } },
      { q: "Open the .env file", args: { path: ".env" } },
      { q: "Read package.json", args: { path: "package.json" } },
      { q: "Show the Gemfile contents", args: { path: "Gemfile" } },
    ]
  },
  write_file: {
    description: "Write content to a file",
    params: { path: "string, required", content: "string, required" },
    examples: [
      { q: "Write 'hello world' to output.txt", args: { path: "output.txt", content: "hello world" } },
      { q: "Create a file test.txt with 'test content'", args: { path: "test.txt", content: "test content" } },
      { q: "Save 'done' to status.txt", args: { path: "status.txt", content: "done" } },
      { q: "Write my notes to notes.md", args: { path: "notes.md", content: "My notes" } },
      { q: "Create log.txt with 'Started'", args: { path: "log.txt", content: "Started" } },
    ]
  },
  run_command: {
    description: "Run a shell command",
    params: { command: "string, required" },
    examples: [
      { q: "Run the tests", args: { command: "npm test" } },
      { q: "Install dependencies", args: { command: "npm install" } },
      { q: "Build the project", args: { command: "npm run build" } },
      { q: "Start the server", args: { command: "npm start" } },
      { q: "Check git status", args: { command: "git status" } },
    ]
  },
  list_files: {
    description: "List files in a directory",
    params: { path: "string, required" },
    examples: [
      { q: "List files in the current directory", args: { path: "." } },
      { q: "Show files in the src folder", args: { path: "src" } },
      { q: "What's in the downloads folder?", args: { path: "downloads" } },
      { q: "List contents of /tmp", args: { path: "/tmp" } },
      { q: "Show files in home directory", args: { path: "~" } },
    ]
  },
  convert_currency: {
    description: "Convert between currencies",
    params: { amount: "number, required", from_currency: "string, required", to_currency: "string, required" },
    examples: [
      { q: "Convert 100 USD to EUR", args: { amount: 100, from_currency: "USD", to_currency: "EUR" } },
      { q: "How much is 50 GBP in JPY?", args: { amount: 50, from_currency: "GBP", to_currency: "JPY" } },
      { q: "Convert 1000 EUR to USD", args: { amount: 1000, from_currency: "EUR", to_currency: "USD" } },
      { q: "What's 200 CAD in AUD?", args: { amount: 200, from_currency: "CAD", to_currency: "AUD" } },
      { q: "Convert 500 CHF to GBP", args: { amount: 500, from_currency: "CHF", to_currency: "GBP" } },
    ]
  },
  get_stock_price: {
    description: "Get current stock price",
    params: { symbol: "string, required" },
    examples: [
      { q: "What's the price of AAPL stock?", args: { symbol: "AAPL" } },
      { q: "Check Google stock price", args: { symbol: "GOOGL" } },
      { q: "Get Tesla stock quote", args: { symbol: "TSLA" } },
      { q: "Price of Microsoft stock", args: { symbol: "MSFT" } },
      { q: "How is Amazon stock doing?", args: { symbol: "AMZN" } },
    ]
  },
  get_directions: {
    description: "Get directions between locations",
    params: { origin: "string, required", destination: "string, required" },
    examples: [
      { q: "Get directions from New York to Boston", args: { origin: "New York", destination: "Boston" } },
      { q: "How do I get from LA to San Diego?", args: { origin: "Los Angeles", destination: "San Diego" } },
      { q: "Directions from Seattle to Portland", args: { origin: "Seattle", destination: "Portland" } },
      { q: "Route from Chicago to Detroit", args: { origin: "Chicago", destination: "Detroit" } },
      { q: "Navigate from Miami to Orlando", args: { origin: "Miami", destination: "Orlando" } },
    ]
  }
}

def format_tool_description(name, tool)
  desc = "#{name}: #{tool[:description]}\n"
  desc += "  Parameters: #{tool[:params].map { |k, v| "#{k} (#{v})" }.join(", ")}"
  desc
end

def generate_output(name, args)
  {
    role: "assistant",
    tool_calls: [
      {
        type: "function",
        function: {
          index: 0,
          name: name.to_s,
          arguments: args
        }
      }
    ]
  }
end

def generate_example(tool_name, tool, example)
  input = "You have access to the following tools:\n\n"
  input += format_tool_description(tool_name, tool)
  input += "\n\nRespond with a JSON tool call if a tool is needed."

  {
    instruction: example[:q],
    input: input,
    output: generate_output(tool_name, example[:args]).to_json
  }
end

# Generate all examples
examples = []

TOOLS.each do |tool_name, tool|
  tool[:examples].each do |example|
    examples << generate_example(tool_name, tool, example)
  end
end

# Shuffle for variety
examples.shuffle!

# Write to file
output_path = File.join(__dir__, "ollama_tool_calls_large.jsonl")
File.open(output_path, "w") do |f|
  examples.each do |ex|
    f.puts ex.to_json
  end
end

puts "Generated #{examples.size} training examples to #{output_path}"
