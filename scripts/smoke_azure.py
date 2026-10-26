# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.60", "python-dotenv>=1.0", "httpx>=0.27"]
# ///
"""Smoke test for the Azure services ReGolazo depends on.

Run with: uv run scripts/smoke_azure.py
Checks both chat deployments and Azure AI Speech TTS for each supported language.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

VOICES = {
    "en": ("en-GB", "en-GB-RyanNeural", "And it's a stunning strike from the edge of the box!"),
    "es": ("es-ES", "es-ES-AlvaroNeural", "¡Golazo! Un disparo espectacular desde la frontal del área."),
    "fr": ("fr-FR", "fr-FR-HenriNeural", "Quel but ! Une frappe magnifique depuis l'entrée de la surface."),
}


def check_chat(client: AzureOpenAI, deployment: str) -> None:
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Reply with exactly: ReGolazo online"}],
        max_completion_tokens=500,
    )
    print(f"[chat] {deployment}: {resp.choices[0].message.content!r}")


def check_tts(lang: str) -> None:
    locale, voice, text = VOICES[lang]
    region = os.environ["AZURE_SPEECH_REGION"]
    ssml = (
        f"<speak version='1.0' xml:lang='{locale}'>"
        f"<voice name='{voice}'>{text}</voice></speak>"
    )
    r = httpx.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"],
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        },
        content=ssml.encode(),
        timeout=30,
    )
    r.raise_for_status()
    out = ROOT / "data" / "cache" / f"smoke_tts_{lang}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print(f"[tts] {voice}: {len(r.content)} bytes -> {out.relative_to(ROOT)}")


def main() -> int:
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_AI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    failures = 0
    for dep in (os.environ["AZURE_OPENAI_DEPLOYMENT_MAIN"], os.environ["AZURE_OPENAI_DEPLOYMENT_FAST"]):
        try:
            check_chat(client, dep)
        except Exception as exc:  # report every failure, not just the first
            failures += 1
            print(f"[chat] {dep}: FAILED {exc}")
    for lang in VOICES:
        try:
            check_tts(lang)
        except Exception as exc:
            failures += 1
            print(f"[tts] {lang}: FAILED {exc}")
    print("ALL OK" if failures == 0 else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
