"""
LLM Models Module
Pre-configured LangChain LLM instances with reasoning on/off variants.

Usage:
    from environment.models import claude_opus, gpt_5, gemini_flash
    from environment.models import claude_opus_thinking, gpt_5_reasoning, gpt_5_mini  # reasoning enabled

    response = claude_opus.invoke("Hello!")
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI

# Load environment variables from root .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_KEY = os.getenv("OPENAI_API_AZURE_KEY")
AZURE_ENDPOINT = os.getenv("OPENAI_API_AZURE_ENDPOINT")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# VertexAI credentials path
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GOOGLE_APPLICATION_CREDENTIALS:
    # Set for google-cloud libraries
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        os.path.dirname(__file__), '..', GOOGLE_APPLICATION_CREDENTIALS
    )

PROJECT_ID = "prism-485101"
LOCATION = "us-central1"


# =============================================================================
# OpenAI Direct API - GPT-5
# =============================================================================

# GPT-5 (reasoning off - minimal)
gpt_5 = ChatOpenAI(
    model="gpt-5",
    api_key=OPENAI_API_KEY,
    temperature=0.0,
    reasoning_effort="minimal",
)

# GPT-5 (reasoning on - default)
gpt_5_reasoning = ChatOpenAI(
    model="gpt-5",
    api_key=OPENAI_API_KEY,
    temperature=0.0,
)


# =============================================================================
# Azure OpenAI - GPT-5-mini
# =============================================================================

# GPT-5-mini (reasoning off - minimal)
gpt_5_mini = AzureChatOpenAI(
    azure_deployment="gpt-5-mini",
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-03-01-preview",
    temperature=1.0,
    reasoning_effort="minimal",
)

# GPT-5-mini (reasoning on - default)
gpt_5_mini_reasoning = AzureChatOpenAI(
    azure_deployment="gpt-5-mini",
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-03-01-preview",
    temperature=1.0,
)


# =============================================================================
# Vertex AI - Gemini Models
# =============================================================================

# Gemini 2.5 Flash (reasoning off)
gemini_2_5_flash = ChatVertexAI(
    model="gemini-2.5-flash",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0,
)

# Gemini 2.5 Flash (reasoning on - thinking enabled)
gemini_2_5_flash_thinking = ChatVertexAI(
    model="gemini-2.5-flash",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0,
    thinking_budget=-1,
)

# Gemini 2.5 Pro (reasoning off)
gemini_2_5_pro = ChatVertexAI(
    model="gemini-2.5-pro",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0,
)

# Gemini 2.5 Pro (reasoning on - thinking enabled)
gemini_2_5_pro_thinking = ChatVertexAI(
    model="gemini-2.5-pro",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0,
    thinking_budget=-1,
)

# Gemini 3 Flash Preview (reasoning off) - requires global endpoint
gemini_3_flash = ChatVertexAI(
    model="gemini-3-flash-preview",
    project=PROJECT_ID,
    location="global",
    temperature=0.0,
)

# Gemini 3 Flash Preview (reasoning on - thinking enabled)
gemini_3_flash_thinking = ChatVertexAI(
    model="gemini-3-flash-preview",
    project=PROJECT_ID,
    location="global",
    temperature=0.0,
    thinking_budget=-1,
)

# Gemini 3 Pro Preview (reasoning off) - requires global endpoint
gemini_3_pro = ChatVertexAI(
    model="gemini-3-pro-preview",
    project=PROJECT_ID,
    location="global",
    temperature=0.0,
)

# Gemini 3 Pro Preview (reasoning on - thinking enabled)
gemini_3_pro_thinking = ChatVertexAI(
    model="gemini-3-pro-preview",
    project=PROJECT_ID,
    location="global",
    temperature=0.0,
    thinking_budget=-1,
)


# =============================================================================
# Vertex AI - Claude Models (Anthropic via Model Garden)
# =============================================================================

# Claude Opus 4.5 (reasoning off)
claude_opus = ChatAnthropic(
    model="claude-opus-4-5-20251101",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.0,
)

# Claude Opus 4.5 (reasoning on - extended thinking)
claude_opus_thinking = ChatAnthropic(
    model="claude-opus-4-5-20251101",
    api_key=ANTHROPIC_API_KEY,
    temperature=1.0,  # required for extended thinking
    thinking={"type": "enabled", "budget_tokens": 16384},
)

# Claude Sonnet 4.5 (reasoning off)
claude_sonnet = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.0,
)

# Claude Sonnet 4.5 (reasoning on - extended thinking)
claude_sonnet_thinking = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=ANTHROPIC_API_KEY,
    temperature=1.0,
    thinking={"type": "enabled", "budget_tokens": 10240},
)

# Claude Haiku 4.5 (reasoning off)
claude_haiku = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.0,
)

# Claude Haiku 4.5 (reasoning on - extended thinking)
claude_haiku_thinking = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=ANTHROPIC_API_KEY,
    temperature=1.0,
    thinking={"type": "enabled", "budget_tokens": 8192},
)


# =============================================================================
# Model Collections
# =============================================================================

# All models without reasoning (minimal)
MODELS = {
    "gpt-5": gpt_5,
    "gpt-5-mini": gpt_5_mini,
    "gemini-2.5-flash": gemini_2_5_flash,
    "gemini-2.5-pro": gemini_2_5_pro,
    "gemini-3-flash": gemini_3_flash,
    "gemini-3-pro": gemini_3_pro,
    "claude-opus": claude_opus,
    "claude-sonnet": claude_sonnet,
    "claude-haiku": claude_haiku,
}

# All models with reasoning enabled (default)
MODELS_REASONING = {
    "gpt-5": gpt_5_reasoning,
    "gpt-5-mini": gpt_5_mini_reasoning,
    "gemini-2.5-flash": gemini_2_5_flash_thinking,
    "gemini-2.5-pro": gemini_2_5_pro_thinking,
    "gemini-3-flash": gemini_3_flash_thinking,
    "gemini-3-pro": gemini_3_pro_thinking,
    "claude-opus": claude_opus_thinking,
    "claude-sonnet": claude_sonnet_thinking,
    "claude-haiku": claude_haiku_thinking,
}
