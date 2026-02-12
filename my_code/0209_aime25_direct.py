"""
AIME25 Direct Solver - Reasoning Only (No Tools)
Uses models.py for LLM instances with LangGraph ReAct Agent (no tools).

Usage:
    python aime25_direct.py --model gemini-pro
    python aime25_direct.py --model gpt-5 --num_problems 10 --reasoning
"""

import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.prebuilt import create_react_agent
from environment.models import MODELS, MODELS_REASONING
from environment.aime25.check_answer import check_answer, extract_answer


def load_dataset(dataset_path: str, num_problems: int = None) -> list:
    """Load AIME25 dataset from JSONL file."""
    problems = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    if num_problems and num_problems < len(problems):
        problems = problems[:num_problems]

    return problems


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def create_prompt(problem: str) -> str:
    """Create a prompt for AIME problems."""
    return f"{problem}\n\n{SYSTEM_PROMPT}"


def solve_problem(agent, problem_data: dict) -> dict:
    """Solve a single problem using LangGraph ReAct Agent (no tools)."""
    problem = problem_data['problem']
    answer = problem_data['answer']
    problem_id = problem_data.get('id', 'unknown')

    prompt = create_prompt(problem)

    result = {
        'id': problem_id,
        'problem': problem,
        'expected_answer': answer,
        'model_response': None,
        'extracted_answer': None,
        'is_correct': False,
        'error': None
    }

    try:
        # Use LangGraph ReAct Agent (with no tools)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"recursion_limit": 50}
        )

        # Extract messages from response
        messages = response.get('messages', [])
        full_response = ""

        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                # Handle string content
                if isinstance(content, str):
                    full_response += content + "\n"
                # Handle list content (Gemini 3 format)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            full_response += item['text'] + "\n"

        result['model_response'] = full_response
        extracted = extract_answer(full_response)
        result['extracted_answer'] = extracted
        result['is_correct'] = check_answer(extracted, answer) if extracted else False

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='AIME25 Problem Solver (Reasoning Only)')
    parser.add_argument('--model', type=str, default='gemini-pro',
                        choices=list(MODELS.keys()),
                        help='Model to use for solving')
    parser.add_argument('--reasoning', action='store_true',
                        help='Use reasoning/thinking enabled model')
    parser.add_argument('--num_problems', type=int, default=None,
                        help='Number of problems to solve (default: all)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'environment', 'aime25', 'dataset', 'test.jsonl')
    output_dir = os.path.join(base_dir, 'my_code', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Select model
    model_dict = MODELS_REASONING if args.reasoning else MODELS
    llm = model_dict[args.model]
    model_suffix = "_reasoning" if args.reasoning else ""

    # Create agent with no tools
    agent = create_react_agent(llm, tools=[])

    # Load dataset
    print(f"\n{'='*60}")
    print(f"AIME25 Problem Solver (LangGraph Agent - No Tools)")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")

    problems = load_dataset(dataset_path, args.num_problems)
    print(f"Problems to solve: {len(problems)}")
    print(f"{'='*60}\n")

    # Solve problems
    results = []
    correct_count = 0

    pbar = tqdm(problems, desc=f"Solving with {args.model}", unit="problem")

    for problem_data in pbar:
        result = solve_problem(agent, problem_data)
        results.append(result)

        if result['is_correct']:
            correct_count += 1

        accuracy = correct_count / len(results) * 100
        pbar.set_postfix({
            'correct': correct_count,
            'accuracy': f'{accuracy:.1f}%'
        })

    # Calculate final statistics
    accuracy = correct_count / len(results) * 100

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")
    print(f"Total Problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace('-', '_')
    output_file = os.path.join(output_dir, f'aime25_direct_{model_name}{model_suffix}_{timestamp}.json')

    output_data = {
        'model': args.model,
        'reasoning': args.reasoning,
        'tools': [],  # No tools used (direct reasoning only)
        'num_problems': len(results),
        'correct': correct_count,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
