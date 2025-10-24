# aisuite

[![PyPI](https://img.shields.io/pypi/v/aisuite)](https://pypi.org/project/aisuite/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple, unified interface to multiple Generative AI providers.

`aisuite` makes it easy for developers to interact with multiple Gen-AI services through a standardized interface. Using an interface similar to OpenAI's, `aisuite` supports **chat completions** and **audio transcription**, making it easy to work with the most popular AI providers and compare results. It is a thin wrapper around python client libraries, and allows creators to seamlessly swap out and test different providers without changing their code.

All of the top providers are supported.
Sample list of supported providers include - Anthropic, AWS, Azure, Cerebras, Cohere, Google (Vertex AI & REST API), Groq, HuggingFace, Ollama, Mistral, OpenAI, Sambanova, Watsonx and others.

To maximize stability, `aisuite` uses either the HTTP endpoint or the SDK for making calls to the provider.

## Installation

You can install just the base `aisuite` package, or install a provider's package along with `aisuite`.

This installs just the base package without installing any provider's SDK.

```shell
pip install aisuite
```

This installs aisuite along with anthropic's library.

```shell
pip install 'aisuite[anthropic]'
```

This installs all the provider-specific libraries

```shell
pip install 'aisuite[all]'
```

## Set up

To get started, you will need API Keys for the providers you intend to use. You'll need to
install the provider-specific library either separately or when installing aisuite.

The API Keys can be set as environment variables, or can be passed as config to the aisuite Client constructor.
You can use tools like [`python-dotenv`](https://pypi.org/project/python-dotenv/) or [`direnv`](https://direnv.net/) to set the environment variables manually. Please take a look at the `examples` folder to see usage.

Here is a short example of using `aisuite` to generate chat completion responses from gpt-4o and claude-3-5-sonnet.

Set the API keys.

```shell
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Use the python client.

```python
import aisuite as ai
client = ai.Client()

models = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.75
    )
    print(response.choices[0].message.content)

```

Note that the model name in the create() call uses the format - `<provider>:<model-name>`.
`aisuite` will call the appropriate provider with the right parameters based on the provider value.
For a list of provider values, you can look at the directory - `aisuite/providers/`. The list of supported providers are of the format - `<provider>_provider.py` in that directory. We welcome  providers adding support to this library by adding an implementation file in this directory. Please see section below for how to contribute.

For more examples, check out the `examples` directory where you will find several notebooks that you can run to experiment with the interface.

## Google Gemini REST API

aisuite now supports Google Gemini through both Vertex AI and REST API modes:

### REST API Mode (Recommended for most users)

Simple setup with just an API key:

```python
import aisuite as ai

# Configure with API key
client = ai.Client({
    "google-rest": {
        "api_key": "your-google-api-key"  # or use GOOGLE_API_KEY env var
    }
})

# Use Gemini models
response = client.chat.completions.create(
    model="google-rest:gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Benefits:**
- ✅ Simple setup (API key only)
- ✅ No Google Cloud project required
- ✅ Free tier available
- ✅ No billing configuration needed

### Vertex AI Mode (Enterprise features)

For advanced features like audio transcription:

```python
# Requires Google Cloud project setup
client = ai.Client({
    "google": {
        "project_id": "your-project-id",
        "region": "us-central1",
        "application_credentials": "/path/to/service-account.json"
    }
})

response = client.chat.completions.create(
    model="google:gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

For detailed setup instructions, see the [Google REST API Guide](guides/google_rest.md).

## Adding support for a provider

We have made easy for a provider or volunteer to add support for a new platform.

### Naming Convention for Provider Modules

We follow a convention-based approach for loading providers, which relies on strict naming conventions for both the module name and the class name. The format is based on the model identifier in the form `provider:model`.

- The provider's module file must be named in the format `<provider>_provider.py`.
- The class inside this module must follow the format: the provider name with the first letter capitalized, followed by the suffix `Provider`.

#### Examples

- **Hugging Face**:
  The provider class should be defined as:

  ```python
  class HuggingfaceProvider(BaseProvider)
  ```

  in providers/huggingface_provider.py.
  
- **OpenAI**:
  The provider class should be defined as:

  ```python
  class OpenaiProvider(BaseProvider)
  ```

  in providers/openai_provider.py

This convention simplifies the addition of new providers and ensures consistency across provider implementations.

## Tool Calling

`aisuite` provides a simple abstraction for tool/function calling that works across supported providers. This is in addition to the regular abstraction of passing JSON spec of the tool to the `tools` parameter. The tool calling abstraction makes it easy to use tools with different LLMs without changing your code.

There are two ways to use tools with `aisuite`:

### 1. Manual Tool Handling

This is the default behavior when `max_turns` is not specified.
You can pass tools in the OpenAI tool format:

```python
def will_it_rain(location: str, time_of_day: str):
    """Check if it will rain in a location at a given time today.
    
    Args:
        location (str): Name of the city
        time_of_day (str): Time of the day in HH:MM format.
    """
    return "YES"

tools = [{
    "type": "function",
    "function": {
        "name": "will_it_rain",
        "description": "Check if it will rain in a location at a given time today",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Name of the city"
                },
                "time_of_day": {
                    "type": "string",
                    "description": "Time of the day in HH:MM format."
                }
            },
            "required": ["location", "time_of_day"]
        }
    }
}]

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=tools
)
```

### 2. Automatic Tool Execution

When `max_turns` is specified, you can pass a list of callable Python functions as the `tools` parameter. `aisuite` will automatically handle the tool calling flow:

```python
def will_it_rain(location: str, time_of_day: str):
    """Check if it will rain in a location at a given time today.
    
    Args:
        location (str): Name of the city
        time_of_day (str): Time of the day in HH:MM format.
    """
    return "YES"

client = ai.Client()
messages = [{
    "role": "user",
    "content": "I live in San Francisco. Can you check for weather "
               "and plan an outdoor picnic for me at 2pm?"
}]

# Automatic tool execution with max_turns
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=[will_it_rain],
    max_turns=2  # Maximum number of back-and-forth tool calls
)
print(response.choices[0].message.content)
```

When `max_turns` is specified, `aisuite` will:
1. Send your message to the LLM
2. Execute any tool calls the LLM requests
3. Send the tool results back to the LLM
4. Repeat until the conversation is complete or max_turns is reached

In addition to `response.choices[0].message`, there is an additional field `response.choices[0].intermediate_messages`: which contains the list of all messages including tool interactions used. This can be used to continue the conversation with the model.
For more detailed examples of tool calling, check out the `examples/tool_calling_abstraction.ipynb` notebook.

## Audio Transcription

> **Note:** Audio transcription support is currently under development. The API and features described below are subject to change.

`aisuite` provides audio transcription (speech-to-text) with the same unified interface pattern used for chat completions. Transcribe audio files across multiple providers with consistent code.

### Basic Usage

```python
import aisuite as ai
client = ai.Client()

# Transcribe an audio file
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="meeting.mp3"
)
print(result.text)

# Switch providers without changing your code
result = client.audio.transcriptions.create(
    model="deepgram:nova-2",
    file="meeting.mp3"
)
print(result.text)
```

### Common Parameters

Use OpenAI-style parameters that work across all providers:

```python
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="interview.mp3",
    language="en",           # Specify audio language
    prompt="Technical discussion about AI",  # Context hints
    temperature=0.2          # Sampling temperature (where supported)
)
```

These parameters are automatically mapped to each provider's native format.

### Provider-Specific Features

Each provider offers unique capabilities you can access directly:

**OpenAI Whisper:**
```python
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="speech.mp3",
    response_format="verbose_json",       # Get detailed metadata
    timestamp_granularities=["word"]      # Word-level timestamps
)
```

**Deepgram:**
```python
result = client.audio.transcriptions.create(
    model="deepgram:nova-2",
    file="meeting.mp3",
    punctuate=True,                       # Auto-add punctuation
    diarize=True,                         # Identify speakers
    sentiment=True,                       # Sentiment analysis
    summarize=True                        # Auto-summarization
)
```

**Google Speech-to-Text:**
```python
result = client.audio.transcriptions.create(
    model="google:default",
    file="call.mp3",
    enable_automatic_punctuation=True,
    enable_speaker_diarization=True,
    diarization_speaker_count=2
)
```

**Hugging Face:**
```python
result = client.audio.transcriptions.create(
    model="huggingface:openai/whisper-large-v3",
    file="presentation.mp3",
    return_timestamps="word"                  # Word-level timestamps
)
```

### Streaming Transcription

For real-time or large audio files, use streaming:

```python
async def transcribe_stream():
    stream = client.audio.transcriptions.create_stream_output(
        model="deepgram:nova-2",
        file="long_recording.mp3"
    )

    async for chunk in stream:
        print(chunk.text, end="", flush=True)
        if chunk.is_final:
            print()  # New line for final results

# Run the async function
import asyncio
asyncio.run(transcribe_stream())
```

### Supported Providers

- **OpenAI**: `whisper-1`
- **Deepgram**: `nova-2`, `nova`, `enhanced`, `base`
- **Google**: `default`, `latest_long`, `latest_short`
- **Hugging Face**: `openai/whisper-large-v3`, `openai/whisper-tiny`, `facebook/wav2vec2-base-960h`, `facebook/wav2vec2-large-xlsr-53`

### Installation

Install transcription providers:

```shell
# Install with specific provider
pip install 'aisuite[openai]'      # For OpenAI Whisper
pip install 'aisuite[deepgram]'    # For Deepgram
pip install 'aisuite[google]'      # For Google Speech-to-Text
pip install 'aisuite[huggingface]' # For Hugging Face models

# Install all providers
pip install 'aisuite[all]'
```

Set API keys:

```shell
export OPENAI_API_KEY="your-openai-api-key"
export DEEPGRAM_API_KEY="your-deepgram-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export HF_TOKEN="your-huggingface-token"
```

For more examples and advanced usage, check out `examples/asr_example.ipynb`.

## License

aisuite is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Contributing

If you would like to contribute, please read our [Contributing Guide](https://github.com/andrewyng/aisuite/blob/main/CONTRIBUTING.md) and join our [Discord](https://discord.gg/T6Nvn8ExSb) server!
