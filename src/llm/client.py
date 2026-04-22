from google import genai
from google.genai import types

from utils import logger, config

class LLMClient:
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    def complete(self, system: str, user: str) -> str:
        """Send a Gemini generation request and return the model's reply.

        Raises:
            RuntimeError: If the API returns an empty response.
            Exception: Propagated from the underlying Gemini client.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )

            if not response.text:
                raise RuntimeError("Empty response from the API")
            return response.text

        except Exception as e:
            logger.error("Error during Gemini completion: %s", e)
            raise

    @property
    def model_name(self) -> str:
        """Return the name of the model this client is configured to use."""
        return self.model
