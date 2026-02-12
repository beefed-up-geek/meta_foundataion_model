"""
MATH-500 Direct Solver - Reasoning Only (No Tools)
Uses models.py for LLM instances with LangGraph ReAct Agent (no tools).

Usage:
    python 0211_math500_direct.py --model gemini-pro
    python 0211_math500_direct.py --model gemini-3-flash --reasoning --num_problems 30
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
# Note: MATH-500 directory has hyphen, need to use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "check_answer",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'environment', 'MATH-500', 'check_answer.py')
)
math500_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math500_check)
check_answer = math500_check.check_answer
extract_answer = math500_check.extract_answer


def load_dataset(dataset_path: str, num_problems: int = None) -> list:
    """Load MATH-500 dataset from JSONL file."""
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
    """Create a prompt for MATH problems."""
    return f"{problem}\n\n{SYSTEM_PROMPT}"


def solve_problem(agent, problem_data: dict) -> dict:
    """Solve a single problem using LangGraph ReAct Agent (no tools)."""
    problem = problem_data['problem']
    answer = problem_data['answer']
    problem_id = problem_data.get('unique_id', 'unknown')
    subject = problem_data.get('subject', 'unknown')
    level = problem_data.get('level', 0)

    prompt = create_prompt(problem)

    result = {
        'id': problem_id,
        'subject': subject,
        'level': level,
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
    parser = argparse.ArgumentParser(description='MATH-500 Problem Solver (Reasoning Only)')
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
    dataset_path = os.path.join(base_dir, 'environment', 'MATH-500', 'dataset', 'test.jsonl')
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
    print(f"MATH-500 Problem Solver (LangGraph Agent - No Tools)")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")

    problems = load_dataset(dataset_path, args.num_problems)
    print(f"Problems to solve: {len(problems)}")
    print(f"{'='*60}\n")

    # Solve problems
    results = []
    correct_count = 0

    # Track by subject and level
    subject_stats = {}
    level_stats = {}

    pbar = tqdm(problems, desc=f"Solving with {args.model}", unit="problem")

    for problem_data in pbar:
        result = solve_problem(agent, problem_data)
        results.append(result)

        # Update stats
        subject = result['subject']
        level = result['level']

        if subject not in subject_stats:
            subject_stats[subject] = {'correct': 0, 'total': 0}
        subject_stats[subject]['total'] += 1

        if level not in level_stats:
            level_stats[level] = {'correct': 0, 'total': 0}
        level_stats[level]['total'] += 1

        if result['is_correct']:
            correct_count += 1
            subject_stats[subject]['correct'] += 1
            level_stats[level]['correct'] += 1

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

    # Print by subject
    print(f"\nBy Subject:")
    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        subj_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {subject}: {stats['correct']}/{stats['total']} ({subj_acc:.1f}%)")

    # Print by level
    print(f"\nBy Level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        level_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  Level {level}: {stats['correct']}/{stats['total']} ({level_acc:.1f}%)")

    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace('-', '_')
    output_file = os.path.join(output_dir, f'math500_direct_{model_name}{model_suffix}_{timestamp}.json')

    output_data = {
        'model': args.model,
        'reasoning': args.reasoning,
        'tools': [],  # No tools used (direct reasoning only)
        'num_problems': len(results),
        'correct': correct_count,
        'accuracy': accuracy,
        'subject_stats': subject_stats,
        'level_stats': level_stats,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
