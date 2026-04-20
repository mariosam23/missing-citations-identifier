from openai import OpenAI

from utils import logger, config

class LLMClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=self.BASE_URL, api_key=config.OPEN_ROUTER_API_KEY)

    def complete(self, system: str, user: str) -> str:
        """Send a chat completion request and return the assistant's reply.

        Raises:
            RuntimeError: If the API returns an empty response.
            Exception: Propagated from the underlying OpenAI client.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Empty response from the API")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error during chat completion: {e}")
            raise
    
    def list_models(self) -> list[str]:
        """List available models from OpenRouter (debug utility)."""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error("Failed to list models from OpenRouter: %s", e)
            return []

    @property
    def model_name(self) -> str:
        """Return the name of the model this client is configured to use."""
        return self.model