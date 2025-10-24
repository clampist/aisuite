#!/usr/bin/env python3
"""
Test Google Gemini API with aisuite

This script verifies that Google Gemini API is correctly configured
and can be used with aisuite library.

Prerequisites:
1. Google API Key (GOOGLE_API_KEY environment variable)
2. aisuite library installed
3. google-generativeai library installed
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()


def test_google_api_key():
    """Test if Google API key is configured"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ GOOGLE_API_KEY is configured")
        print(f"   Key: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("❌ GOOGLE_API_KEY is not configured")
        return False


def test_aisuite_installation():
    """Test if aisuite package is installed"""
    print("\n🔍 Checking aisuite Installation")
    print("=" * 60)
    
    try:
        import aisuite as ai
        print("✅ aisuite package is installed")
        return True
    except ImportError:
        print("❌ aisuite package is not installed")
        print("   Install with: pip install aisuite")
        return False


def test_google_generativeai_installation():
    """Test if google-generativeai package is installed"""
    print("\n🔍 Checking google-generativeai Installation")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
        return True
    except ImportError:
        print("❌ google-generativeai package is not installed")
        print("   Install with: pip install google-generativeai")
        return False


def test_direct_genai():
    """Test direct usage of genai client"""
    print("\n🧪 Testing Direct genai Usage")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            contents="Say 'Hello from Gemini 2.5!' in one sentence."
        )
        
        print(f"✅ Direct genai test successful: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Direct genai test failed: {e}")
        return False


def test_google_rest_api_simple():
    """Test Google REST API with a simple chat completion"""
    print("\n🧪 Testing Google REST API with Simple Chat Completion")
    print("=" * 60)
    
    try:
        import aisuite as ai
        
        # Configure client with Google REST API
        client = ai.Client({
            "google_rest": {
                "api_key": os.getenv("GOOGLE_API_KEY")
            }
        })
        
        model = "google_rest:gemini-2.5-flash"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond concisely."},
            {"role": "user", "content": "Say 'Hello from Google REST API!' in one sentence."},
        ]
        
        print(f"📤 Sending request to {model}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        
        print("✅ Successfully received response from Google REST API")
        print(f"📥 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to call Google REST API: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"\n   Traceback:\n{traceback.format_exc()}")
        return False


def test_google_rest_api_with_system_prompt():
    """Test Google REST API with a system prompt"""
    print("\n🧪 Testing Google REST API with System Prompt")
    print("=" * 60)
    
    try:
        import aisuite as ai
        
        client = ai.Client({
            "google_rest": {
                "api_key": os.getenv("GOOGLE_API_KEY")
            }
        })
        
        model = "google_rest:gemini-2.5-flash"
        
        messages = [
            {"role": "system", "content": "Respond in Pirate English."},
            {"role": "user", "content": "Tell me a joke."},
        ]
        
        print(f"📤 Sending request to {model} with Pirate English system prompt...")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8
        )
        
        print("✅ Successfully received response from Google REST API")
        print(f"📥 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to call Google REST API: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_google_rest_api_models():
    """Test different Google REST API models"""
    print("\n🧪 Testing Different Google REST API Models")
    print("=" * 60)
    
    models = [
        "google_rest:gemini-2.5-flash",
    ]
    
    results = {}
    
    for model in models:
        print(f"\n📤 Testing {model}...")
        try:
            import aisuite as ai
            
            client = ai.Client({
                "google_rest": {
                    "api_key": os.getenv("GOOGLE_API_KEY")
                }
            })
            
            messages = [
                {"role": "user", "content": "Say hello in one word."},
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5
            )
            
            print(f"   ✅ {model}: {response.choices[0].message.content}")
            results[model] = True
            
        except Exception as e:
            print(f"   ❌ {model}: {e}")
            results[model] = False
    
    return all(results.values())


def test_google_rest_api_tool_calling():
    """Test Google REST API with tool calling"""
    print("\n🧪 Testing Google REST API with Tool Calling")
    print("=" * 60)
    
    try:
        import aisuite as ai
        from aisuite.utils.tools import Tools
        
        # Define a simple tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"The weather in {location} is sunny and 72°F."
        
        tools = Tools([get_weather])
        
        client = ai.Client({
            "google_rest": {
                "api_key": os.getenv("GOOGLE_API_KEY")
            }
        })
        
        model = "google_rest:gemini-2.5-flash"
        
        messages = [
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ]
        
        print(f"📤 Sending request to {model} with tool calling...")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_turns=2
        )
        
        print("✅ Successfully received response from Google REST API with tool calling")
        print(f"📥 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to call Google REST API with tool calling: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"\n   Traceback:\n{traceback.format_exc()}")
        return False


def print_setup_instructions():
    """Print setup instructions for Google REST API"""
    print("\n" + "=" * 60)
    print("📚 Google Gemini REST API Setup Instructions")
    print("=" * 60)
    print("""
1. Get a Google API Key:
   - Visit https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy the API key

2. Set Environment Variable:
   export GOOGLE_API_KEY=your-api-key-here
   
   Or add to .env file:
   GOOGLE_API_KEY=your-api-key-here

3. Install Required Packages:
   pip install aisuite google-generativeai python-dotenv

4. Run this test again to verify setup

💡 Benefits of REST API over Vertex AI:
   - No Google Cloud project setup required
   - No billing configuration needed
   - Simpler authentication (just API key)
   - Free tier available
""")


def main():
    print("🧪 Google Gemini API Configuration Test (with aisuite)")
    print("=" * 60)
    
    # Check API key
    api_key_ok = test_google_api_key()
    
    # Check package installations
    aisuite_ok = test_aisuite_installation()
    genai_ok = test_google_generativeai_installation()
    
    # If prerequisites are not met, show setup instructions
    if not (api_key_ok and aisuite_ok and genai_ok):
        print("\n⚠️  Prerequisites not met. Please complete the setup first.")
        print_setup_instructions()
        return 1
    
    # Run tests
    direct_genai_ok = test_direct_genai()
    simple_test_ok = test_google_rest_api_simple()
    system_prompt_test_ok = test_google_rest_api_with_system_prompt()
    models_test_ok = test_google_rest_api_models()
    tool_calling_test_ok = test_google_rest_api_tool_calling()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Google API Key: {'✅ PASS' if api_key_ok else '❌ FAIL'}")
    print(f"aisuite Library: {'✅ PASS' if aisuite_ok else '❌ FAIL'}")
    print(f"google-generativeai Library: {'✅ PASS' if genai_ok else '❌ FAIL'}")
    print(f"Direct genai Test: {'✅ PASS' if direct_genai_ok else '❌ FAIL'}")
    print(f"Simple Chat Completion: {'✅ PASS' if simple_test_ok else '❌ FAIL'}")
    print(f"System Prompt Test: {'✅ PASS' if system_prompt_test_ok else '❌ FAIL'}")
    print(f"Multiple Models Test: {'✅ PASS' if models_test_ok else '❌ FAIL'}")
    print(f"Tool Calling Test: {'✅ PASS' if tool_calling_test_ok else '❌ FAIL'}")
    
    if all([api_key_ok, aisuite_ok, genai_ok, direct_genai_ok, simple_test_ok, system_prompt_test_ok, models_test_ok, tool_calling_test_ok]):
        print("\n🎉 All tests passed! Google Gemini API is ready to use with aisuite.")
        print("\n💡 You can now use Google REST API in your agents:")
        print("   Set ACTIVE_MODEL=google_rest:gemini-2.5-flash in .env")
        print("   Or use model='google_rest:gemini-2.5-flash' in your code")
        print("\n🆚 Comparison with Vertex AI:")
        print("   ✅ Simpler setup (just API key)")
        print("   ✅ No billing required")
        print("   ✅ Free tier available")
        print("   ✅ No Google Cloud project needed")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        
        if not api_key_ok:
            print_setup_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
