"""
LLM-as-Judge answer checker for BrowseComp-Plus dataset.
Uses Qwen3-32B via OpenRouter API with enable_thinking=False.

Based on the original BrowseComp-Plus evaluation logic.

Usage:
    from check_answer import check_answer
    result = check_answer(question, response, correct_answer)

    # Or from command line:
    python check_answer.py --question "What is 2+2?" --response "The answer is 4" --answer "4"

Environment:
    OPENROUTER_API_KEY: Your OpenRouter API key
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from current directory or parent directories
load_dotenv()
# Also try to load from the environment directory
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Original GRADER_TEMPLATE from BrowseComp-Plus
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()

# Default model settings from original BrowseComp-Plus
DEFAULT_MODEL = "qwen/qwen3-32b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 4096


def get_client() -> OpenAI:
    """Get OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it with your OpenRouter API key."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Create the judge prompt from template."""
    return GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def parse_judge_response(judge_response: str) -> Dict[str, Any]:
    """
    Parse the judge model's response to extract structured results.

    Returns:
        dict with keys:
            - extracted_final_answer: str or None
            - reasoning: str or None
            - correct: bool or None
            - confidence: float or None
            - parse_error: bool
    """
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning/explanation
    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract correct (yes/no)
    correct_match = re.search(
        r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE
    )
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE
        )
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    # Extract confidence (percentage)
    confidence_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
    )
    if not confidence_match:
        confidence_match = re.search(
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if not confidence_match:
        confidence_match = re.search(
            r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    # Check if we got the essential fields
    if result["correct"] is None:
        result["parse_error"] = True

    return result


def call_judge_model(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Call the judge model via OpenRouter API.

    Args:
        prompt: The judge prompt
        model: Model ID (default: qwen/qwen3-32b)
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
        top_k: Top-k sampling
        max_tokens: Maximum output tokens

    Returns:
        The model's response text
    """
    client = get_client()

    # For Qwen3 models, we need to disable thinking mode
    # OpenRouter supports this via extra_body or provider-specific parameters
    extra_body = {
        "top_k": top_k,
        # Disable thinking for Qwen3 models
        "provider": {
            "order": ["Together"],  # Use Together AI as provider for Qwen3
            "allow_fallbacks": True,
        },
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that judges whether answers are correct. Do not use thinking tags or extended reasoning. Respond directly with the requested format."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling judge model: {e}")
        return ""


def check_answer(
    question: str,
    response: str,
    correct_answer: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    return_details: bool = False,
) -> bool | Dict[str, Any]:
    """
    Check if a response correctly answers the question.

    Uses Qwen3-32B as LLM-judge via OpenRouter API.

    Args:
        question: The original question
        response: The model's response to judge
        correct_answer: The ground truth answer
        model: Judge model ID
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        max_tokens: Max output tokens
        return_details: If True, return full judge result dict instead of bool

    Returns:
        bool: True if correct, False otherwise (when return_details=False)
        dict: Full judge result with reasoning (when return_details=True)

    Example:
        >>> check_answer(
        ...     question="What is the capital of France?",
        ...     response="The capital of France is Paris.",
        ...     correct_answer="Paris"
        ... )
        True
    """
    # Handle empty/None inputs
    if not response or response.strip() == "":
        if return_details:
            return {
                "correct": False,
                "extracted_final_answer": None,
                "reasoning": "Empty response",
                "confidence": None,
                "parse_error": True,
                "judge_response": None,
            }
        return False

    # Create judge prompt
    judge_prompt = create_judge_prompt(question, response, correct_answer)

    # Call judge model
    judge_response = call_judge_model(
        prompt=judge_prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    # Parse result
    result = parse_judge_response(judge_response)
    result["judge_response"] = judge_response

    if return_details:
        return result

    return result.get("correct", False)


def check_answer_batch(
    items: list[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Dict[str, Any]]:
    """
    Check multiple answers in batch.

    Args:
        items: List of dicts with keys: question, response, correct_answer
        model: Judge model ID
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        max_tokens: Max output tokens

    Returns:
        List of judge result dicts
    """
    results = []
    for item in items:
        result = check_answer(
            question=item["question"],
            response=item["response"],
            correct_answer=item["correct_answer"],
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            return_details=True,
        )
        result["query_id"] = item.get("query_id")
        results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if a response correctly answers a question using LLM-as-Judge"
    )
    parser.add_argument("--question", "-q", required=True, help="The question")
    parser.add_argument("--response", "-r", required=True, help="The response to judge")
    parser.add_argument("--answer", "-a", required=True, help="The correct answer")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature"
    )
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Top-p")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-k")
    parser.add_argument(
        "--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens"
    )
    parser.add_argument(
        "--details", action="store_true", help="Show detailed judge response"
    )
    args = parser.parse_args()

    result = check_answer(
        question=args.question,
        response=args.response,
        correct_answer=args.answer,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        return_details=True,
    )

    print(f"Question: {args.question}")
    print(f"Response: {args.response}")
    print(f"Correct Answer: {args.answer}")
    print(f"\n{'='*50}")
    print(f"Correct: {result.get('correct')}")
    print(f"Extracted Answer: {result.get('extracted_final_answer')}")
    print(f"Confidence: {result.get('confidence')}%")

    if args.details:
        print(f"\n{'='*50}")
        print(f"Reasoning: {result.get('reasoning')}")
        print(f"\n{'='*50}")
        print(f"Full Judge Response:\n{result.get('judge_response')}")
