"""The interface to Google's Gemini REST API."""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Union, BinaryIO, AsyncGenerator

from aisuite.framework import ChatCompletionResponse, Message
from aisuite.framework.message import (
    TranscriptionResult,
    Word,
    Segment,
    Alternative,
    StreamingTranscriptionChunk,
)
from aisuite.provider import Provider, ASRError, Audio


DEFAULT_TEMPERATURE = 0.7
ENABLE_DEBUG_MESSAGES = False

# Google Gemini REST API endpoints
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GoogleRestMessageConverter:
    """Convert messages between aisuite format and Google Gemini REST API format."""
    
    @staticmethod
    def convert_request(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert aisuite messages to Google Gemini REST API format."""
        # Convert all messages to dicts if they're Message objects
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message
            for message in messages
        ]

        contents = []
        system_instruction = None
        
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # System messages become system instruction
                system_instruction = content
            elif role == "user":
                # User messages
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                # Assistant messages
                if "tool_calls" in message and message["tool_calls"]:
                    # Handle function calls
                    for tool_call in message["tool_calls"]:
                        function_call = tool_call["function"]
                        contents.append({
                            "role": "model",
                            "parts": [{
                                "function_call": {
                                    "name": function_call["name"],
                                    "args": json.loads(function_call["arguments"])
                                }
                            }]
                        })
                else:
                    # Regular text response
                    contents.append({
                        "role": "model", 
                        "parts": [{"text": content}]
                    })
            elif role == "tool":
                # Tool responses
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": message["name"],
                            "response": json.loads(content) if isinstance(content, str) else content
                        }
                    }]
                })

        request_data = {"contents": contents}
        if system_instruction:
            request_data["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        return request_data

    @staticmethod
    def convert_response(response_data: Dict[str, Any]) -> ChatCompletionResponse:
        """Convert Google Gemini REST API response to aisuite format."""
        aisuite_response = ChatCompletionResponse()
        
        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the response")
            print(json.dumps(response_data, indent=2))

        try:
            candidate = response_data["candidates"][0]
            content = candidate["content"]
            parts = content["parts"]
            
            # Check if response contains function calls
            function_calls = []
            text_content = ""
            
            for part in parts:
                if "function_call" in part:
                    function_call = part["function_call"]
                    function_calls.append({
                        "type": "function",
                        "id": f"call_{hash(function_call['name'])}",
                        "function": {
                            "name": function_call["name"],
                            "arguments": json.dumps(function_call["args"])
                        }
                    })
                elif "text" in part:
                    text_content += part["text"]
            
            if function_calls:
                # Function call response
                aisuite_response.choices[0].message = Message(
                    role="assistant",
                    content=None,
                    tool_calls=function_calls
                )
                aisuite_response.choices[0].finish_reason = "tool_calls"
            else:
                # Regular text response
                aisuite_response.choices[0].message = Message(
                    role="assistant",
                    content=text_content
                )
                aisuite_response.choices[0].finish_reason = "stop"
                
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format from Google Gemini API: {e}")
            
        return aisuite_response


class GoogleRestProvider(Provider):
    """Implements the Provider interface for Google's Gemini REST API."""

    def __init__(self, **config):
        """Initialize the Google REST API client."""
        super().__init__()
        
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is required for Google REST API. "
                "Set it in environment variables or provider config."
            )
        
        self.transformer = GoogleRestMessageConverter()
        
        # Initialize audio functionality (placeholder for now)
        self.audio = GoogleRestAudio(self)

    def chat_completions_create(self, model, messages, **kwargs):
        """Request chat completions from Google Gemini REST API.

        Args:
        ----
            model (str): The model name (e.g., "gemini-2.0-flash-exp").
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional arguments for the API.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.
        """
        
        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        
        # Convert messages to Gemini REST API format
        request_data = self.transformer.convert_request(messages)
        
        # Add generation config
        request_data["generationConfig"] = {
            "temperature": temperature,
            "maxOutputTokens": kwargs.get("max_tokens", 8192),
            "topP": kwargs.get("top_p", 0.95),
            "topK": kwargs.get("top_k", 40)
        }
        
        # Handle tools if provided
        if "tools" in kwargs:
            tools = []
            for tool in kwargs["tools"]:
                tools.append({
                    "function_declarations": [{
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"]
                    }]
                })
            request_data["tools"] = tools
        
        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the request data")
            print(json.dumps(request_data, indent=2))
        
        # Make the API call
        url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
        params = {"key": self.api_key}
        
        try:
            response = requests.post(url, json=request_data, params=params, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            return self.transformer.convert_response(response_data)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to call Google Gemini REST API: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Google Gemini API: {e}")


class GoogleRestAudio(Audio):
    """Google REST API Audio functionality container."""

    def __init__(self, provider):
        super().__init__()
        self.provider = provider
        self.transcriptions = self.Transcriptions(provider)

    class Transcriptions(Audio.Transcription):
        """Google REST API Audio Transcriptions functionality."""

        def __init__(self, provider):
            self.provider = provider

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> TranscriptionResult:
            """
            Create audio transcription using Google Gemini API.
            
            Note: Google Gemini REST API doesn't support audio transcription directly.
            This is a placeholder implementation.
            """
            raise NotImplementedError(
                "Audio transcription is not supported by Google Gemini REST API. "
                "Use Google Vertex AI provider for audio transcription."
            )

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """
            Create streaming audio transcription.
            
            Note: Google Gemini REST API doesn't support audio transcription directly.
            This is a placeholder implementation.
            """
            raise NotImplementedError(
                "Audio transcription is not supported by Google Gemini REST API. "
                "Use Google Vertex AI provider for audio transcription."
            )
