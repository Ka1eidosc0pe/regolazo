"""Azure OpenAI (Foundry) clients for Microsoft Agent Framework, and Azure Speech text-to-speech.

Keys come from the main checkout's ``.env`` (loaded by absolute path, never printed). Findings from
docs/research/agent_framework.md: pass ``azure_endpoint`` and ``api_key`` explicitly; gpt-5-mini rejects
``temperature``; cap output with ``max_tokens`` (sent as ``max_completion_tokens``, hidden reasoning included).
"""

import asyncio
import os
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

ENV_PATH = Path("/Users/adityaselvam/Desktop/Aage/Hackathons/PL_MSFT_InsideTheGame_2026/.env")
MAIN_MODEL = "gpt-5-mini"
FAST_MODEL = "gpt-4.1-mini"

VOICES = {
    "en": ("en-GB", "en-GB-RyanNeural"),
    "es": ("es-ES", "es-ES-AlvaroNeural"),
    "fr": ("fr-FR", "fr-FR-HenriNeural"),
}


def load_env() -> None:
    """Load Azure settings from the main checkout ``.env`` (no-op if already in the environment)."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)


def make_client(deployment: str) -> OpenAIChatCompletionClient:
    load_env()
    client = OpenAIChatCompletionClient(
        model=deployment,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_AI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    client.function_invocation_configuration["max_iterations"] = 4
    client.function_invocation_configuration["max_function_calls"] = 4
    return client


def make_agent(deployment: str, name: str, instructions: str, tools: list[object] | None = None) -> Agent:
    """One agent on one deployment. gpt-5-mini gets no temperature and a reasoning-aware token cap."""
    options: dict[str, object] = {"max_tokens": 4000}
    if deployment == MAIN_MODEL:
        options["reasoning_effort"] = "low"
    else:
        options["max_tokens"] = 800
    return Agent(
        client=make_client(deployment),
        name=name,
        instructions=instructions,
        tools=tools,
        default_options=options,  # type: ignore[arg-type]
    )


async def tts(text: str, language: str, voice: str | None = None, attempts: int = 3) -> bytes:
    """Synthesize ``text`` with an Azure neural voice; returns mp3 bytes (24 kHz, 48 kbps mono)."""
    load_env()
    locale, default_voice = VOICES[language]
    ssml = (
        f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice or default_voice}'>"
        f"{escape(text)}</voice></speak>"
    )
    url = f"https://{os.environ['AZURE_SPEECH_REGION']}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"],
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
    }
    last: Exception | None = None
    async with httpx.AsyncClient(timeout=30) as http:
        for k in range(attempts):
            try:
                r = await http.post(url, headers=headers, content=ssml.encode())
                r.raise_for_status()
                return r.content
            except httpx.HTTPError as exc:
                last = exc
                await asyncio.sleep(1.5 * (k + 1))
    assert last is not None
    raise last
