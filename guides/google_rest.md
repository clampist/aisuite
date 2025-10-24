# Google Gemini REST API Guide

This guide explains how to use Google Gemini REST API with aisuite, providing a simpler alternative to Vertex AI.

## 🚀 Quick Start

### 1. Get API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY=your-api-key-here
```

Or add to your `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

### 3. Install Dependencies

```bash
pip install aisuite requests
```

### 4. Basic Usage

```python
import aisuite as ai

# Configure client
client = ai.Client({
    "google-rest": {
        "api_key": "your-api-key-here"  # or use environment variable
    }
})

# Simple chat completion
response = client.chat.completions.create(
    model="google-rest:gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## 📋 Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `google-rest:gemini-2.5-flash` | Latest fast model | Best performance |


## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google API key | Yes |

### Provider Configuration

```python
client = ai.Client({
    "google-rest": {
        "api_key": "your-api-key-here"  # Optional if GOOGLE_API_KEY is set
    }
})
```

## 💬 Usage Examples

### Basic Chat Completion

```python
import aisuite as ai

client = ai.Client()

response = client.chat.completions.create(
    model="google-rest:gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I create a Python function?"},
]

response = client.chat.completions.create(
    model="google-rest:gemini-2.5-flash",
    messages=messages
)

# Continue conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Can you show me an example?"})

response = client.chat.completions.create(
    model="google-rest:gemini-2.5-flash",
    messages=messages
)
```

### Tool Calling

```python
from aisuite.utils.tools import Tools

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny and 72°F."

tools = Tools([get_weather])

response = client.chat.completions.create(
    model="google-rest:gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=tools,
    max_turns=2
)
```

## 🆚 REST API vs Vertex AI

| Feature | REST API | Vertex AI |
|---------|----------|-----------|
| **Setup Complexity** | Simple (API key only) | Complex (project, billing, credentials) |
| **Authentication** | API key | Service account JSON |
| **Billing** | Optional (free tier) | Required |
| **Google Cloud Project** | Not needed | Required |
| **Audio Transcription** | Not supported | Supported |
| **Enterprise Features** | Limited | Full |
| **Cost** | Free tier available | Pay-per-use |

## 🔍 Troubleshooting

### Common Issues

#### 1. API Key Not Found

```
Error: GOOGLE_API_KEY is required for Google REST API
```

**Solution**: Set your API key:
```bash
export GOOGLE_API_KEY=your-api-key-here
```

#### 2. Invalid Model Format

```
Error: Invalid model format. Expected 'provider:model'
```

**Solution**: Use correct format:
```python
model="google-rest:gemini-2.5-flash"  # ✅ Correct
model="gemini-2.5-flash"              # ❌ Wrong
```

#### 3. Module Import Error

```
Error: Could not import module aisuite.providers.google_rest_provider
```

**Solution**: Ensure you have the latest aisuite version:
```bash
pip install --upgrade aisuite
```

### Debug Mode

Enable debug messages to see request/response details:

```python
# In google_rest_provider.py, set:
ENABLE_DEBUG_MESSAGES = True
```

## 📊 Performance Tips

1. **Use appropriate models**:
   - `gemini-1.5-flash` for speed
   - `gemini-1.5-pro` for quality
   - `gemini-2.5-flash` for latest features

2. **Optimize temperature**:
   - `0.0-0.3`: Deterministic responses
   - `0.7`: Balanced creativity
   - `1.0`: Maximum creativity

3. **Batch requests** when possible to reduce API calls

## 🔒 Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for API keys
3. **Rotate API keys** regularly
4. **Monitor usage** through Google AI Studio

## 📚 Additional Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [aisuite Documentation](../README.md)
- [Examples](../examples/)

## 🆘 Support

If you encounter issues:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [examples](../examples/)
3. Open an issue on the aisuite repository
