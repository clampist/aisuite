"""The interface to Google's Gemini REST API using the new genai client."""

import os
import json
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


class GoogleRestMessageConverter:
    """Convert messages between aisuite format and Google Gemini genai format."""
    
    @staticmethod
    def convert_request(messages: List[Dict[str, Any]]) -> tuple[str, Optional[str]]:
        """Convert aisuite messages to Google Gemini genai format.
        
        Returns:
            tuple: (contents, system_instruction)
        """
        # Convert all messages to dicts if they're Message objects
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message
            for message in messages
        ]

        contents = ""
        system_instruction = None
        
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # System messages become system instruction
                system_instruction = content
            elif role == "user":
                # User messages
                if contents:
                    contents += f"\n\nUser: {content}"
                else:
                    contents = f"User: {content}"
            elif role == "assistant":
                # Assistant messages
                contents += f"\n\nAssistant: {content}"
            elif role == "tool":
                # Tool responses
                contents += f"\n\nTool Response: {content}"

        return contents, system_instruction

    @staticmethod
    def convert_response(response) -> ChatCompletionResponse:
        """Convert Google Gemini genai response to aisuite format."""
        aisuite_response = ChatCompletionResponse()
        
        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the response")
            print(f"Response type: {type(response)}")
            print(f"Response: {response}")

        try:
            # Extract text content from response
            if hasattr(response, 'text'):
                text_content = response.text
            elif hasattr(response, 'content'):
                text_content = response.content
            else:
                text_content = str(response)
            
            # Check if response contains function calls (tools)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    for part in parts:
                        if hasattr(part, 'function_call'):
                            # Handle function calls
                            function_call = part.function_call
                            function_calls = [{
                                "type": "function",
                                "id": f"call_{hash(function_call.name)}",
                                "function": {
                                    "name": function_call.name,
                                    "arguments": json.dumps(dict(function_call.args))
                                }
                            }]
                            
                            aisuite_response.choices[0].message = Message(
                                role="assistant",
                                content=None,
                                tool_calls=function_calls
                            )
                            aisuite_response.choices[0].finish_reason = "tool_calls"
                            return aisuite_response
            
            # Regular text response
            aisuite_response.choices[0].message = Message(
                role="assistant",
                content=text_content
            )
            aisuite_response.choices[0].finish_reason = "stop"
                
        except Exception as e:
            # Fallback for simple text response
            text_content = str(response) if not hasattr(response, 'text') else response.text
            aisuite_response.choices[0].message = Message(
                role="assistant",
                content=text_content
            )
            aisuite_response.choices[0].finish_reason = "stop"
            
        return aisuite_response


class GoogleRestProvider(Provider):
    """Implements the Provider interface for Google's Gemini REST API using genai client."""

    def __init__(self, **config):
        """Initialize the Google REST API client using genai."""
        super().__init__()
        
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is required for Google REST API. "
                "Set it in environment variables or provider config."
            )
        
        # Import and configure genai
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Google REST API. "
                "Install it with: pip install google-generativeai"
            )
        
        self.transformer = GoogleRestMessageConverter()
        
        # Initialize audio functionality (placeholder for now)
        self.audio = GoogleRestAudio(self)

    def chat_completions_create(self, model, messages, **kwargs):
        """Request chat completions from Google Gemini REST API using genai client.

        Args:
        ----
            model (str): The model name (e.g., "gemini-2.5-flash").
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional arguments for the API.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.
        """
        
        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        
        # Convert messages to genai format
        contents, system_instruction = self.transformer.convert_request(messages)
        
        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the request data")
            print(f"Contents: {contents}")
            print(f"System instruction: {system_instruction}")
        
        try:
            # Create GenerativeModel
            model_instance = self.genai.GenerativeModel(model)
            
            # Prepare generation config
            generation_config = self.genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=kwargs.get("max_tokens", 8192),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 40)
            )
            
            # Generate content
            # Note: system_instruction is not supported in the current API
            # We'll include it in the contents instead
            if system_instruction:
                full_contents = f"System: {system_instruction}\n\n{contents}"
            else:
                full_contents = contents
                
            response = model_instance.generate_content(
                contents=full_contents,
                generation_config=generation_config
            )
            
            return self.transformer.convert_response(response)
            
        except Exception as e:
            raise ValueError(f"Failed to call Google Gemini REST API: {e}")


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