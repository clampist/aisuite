"""
Google Gemini REST API Usage Examples with aisuite

This script demonstrates how to use Google Gemini REST API with aisuite.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()


def basic_usage_example():
    """Basic usage example with Google REST API."""
    print("🚀 Basic Usage Example")
    print("=" * 50)
    
    import aisuite as ai
    
    # Configure client with Google REST API
    client = ai.Client({
        "google-rest": {
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    })
    
    # Simple chat completion
    response = client.chat.completions.create(
        model="google-rest:gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "Hello! Can you tell me a joke?"}
        ],
        temperature=0.7
    )
    
    print(f"Response: {response.choices[0].message.content}")


def conversation_example():
    """Multi-turn conversation example."""
    print("\n💬 Conversation Example")
    print("=" * 50)
    
    import aisuite as ai
    
    client = ai.Client({
        "google-rest": {
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    })
    
    # Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    
    response = client.chat.completions.create(
        model="google-rest:gemini-2.5-flash",
        messages=messages,
        temperature=0.5
    )
    
    print(f"Assistant: {response.choices[0].message.content}")
    
    # Continue conversation
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    messages.append({"role": "user", "content": "Can you give me a simple example?"})
    
    response = client.chat.completions.create(
        model="google-rest:gemini-2.5-flash",
        messages=messages,
        temperature=0.5
    )
    
    print(f"Assistant: {response.choices[0].message.content}")


def tool_calling_example():
    """Tool calling example."""
    print("\n🔧 Tool Calling Example")
    print("=" * 50)
    
    import aisuite as ai
    from aisuite.utils.tools import Tools
    
    # Define tools
    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        return f"The weather in {location} is sunny and 75°F."
    
    def get_time() -> str:
        """Get current time."""
        import datetime
        return f"Current time is {datetime.datetime.now().strftime('%H:%M:%S')}"
    
    tools = Tools([get_weather, get_time])
    
    client = ai.Client({
        "google-rest": {
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    })
    
    response = client.chat.completions.create(
        model="google-rest:gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo and what time is it?"}
        ],
        tools=tools,
        max_turns=3
    )
    
    print(f"Final Response: {response.choices[0].message.content}")


def model_comparison_example():
    """Compare different Gemini models."""
    print("\n🔄 Model Comparison Example")
    print("=" * 50)
    
    import aisuite as ai
    
    client = ai.Client({
        "google-rest": {
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    })
    
    models = [
        "google-rest:gemini-1.5-flash",
        "google-rest:gemini-1.5-pro",
        "google-rest:gemini-2.5-flash"
    ]
    
    question = "Explain quantum computing in one sentence."
    
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.3
            )
            print(f"\n{model}:")
            print(f"  {response.choices[0].message.content}")
        except Exception as e:
            print(f"\n{model}: ❌ Error - {e}")


def main():
    """Run all examples."""
    print("🧪 Google Gemini REST API Usage Examples")
    print("=" * 60)
    
    # Check if API key is configured
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY is not configured")
        print("Please set your Google API key in the environment or .env file")
        return 1
    
    try:
        basic_usage_example()
        conversation_example()
        tool_calling_example()
        model_comparison_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Key Benefits of Google REST API:")
        print("   ✅ Simple setup (just API key)")
        print("   ✅ No Google Cloud project required")
        print("   ✅ No billing configuration needed")
        print("   ✅ Free tier available")
        print("   ✅ Direct API access")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
