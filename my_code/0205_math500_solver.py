"""
MATH-500 Problem Solver using LangGraph ReAct Agent
Supports multi-hop tool calling with GPT, Gemini, Claude, Kimi models.

Usage:
    python 2026-02-05_math500_solver.py --model gpt
    python 2026-02-05_math500_solver.py --model claude --num_problems 100
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.Agents import get_agent, get_llm, AVAILABLE_MODELS
import importlib.util

# Import from MATH-500 (handle hyphen in folder name)
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools_module = import_from_path("tools", os.path.join(base_dir, "environment", "MATH-500", "agent_tools", "tools.py"))
check_module = import_from_path("check_answer", os.path.join(base_dir, "environment", "MATH-500", "check_answer.py"))

math_tools = tools_module.math_tools
check_answer = check_module.check_answer


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


def create_prompt(problem: str, subject: str = None, level: int = None) -> str:
    """Create a prompt for the agent to solve the problem."""
    context = ""
    if subject:
        context += f"Subject: {subject}\n"
    if level:
        context += f"Difficulty Level: {level}/5\n"

    return f"""You are an expert mathematician solving competition-level math problems.

{context}
PROBLEM:
{problem}

INSTRUCTIONS:
1. Read the problem carefully and identify what is being asked.
2. Use the available tools (python_repl, sympy_calculator) to perform ALL calculations - do not calculate in your head.
3. For problems with diagrams or graphs described in [asy] code, carefully analyze the geometry/coordinates.
4. Double-check your answer before finalizing.
5. Provide your answer in the simplest form (reduced fractions, simplified radicals, etc.).

CRITICAL RULES:
- ALWAYS use tools for calculations, especially for arithmetic and algebraic manipulation.
- For counting problems, enumerate all possibilities systematically.
- For geometry problems, set up coordinates or use geometric relationships carefully.
- Verify your answer satisfies the original problem conditions.

YOUR FINAL ANSWER MUST BE IN THIS EXACT FORMAT:
**FINAL ANSWER: [answer]**

IMPORTANT: Put ONLY the final numerical or algebraic answer after "FINAL ANSWER:".
- Do NOT use \\boxed{{}}
- Do NOT include units unless the problem asks for them
- Use LaTeX for fractions (\\frac{{a}}{{b}}), square roots (\\sqrt{{n}}), etc.

Examples:
- **FINAL ANSWER: 42**
- **FINAL ANSWER: \\frac{{1}}{{2}}**
- **FINAL ANSWER: 3\\sqrt{{13}}**
- **FINAL ANSWER: (3, 4)**

Begin solving:"""


def extract_answer(response: str) -> str:
    """Extract the final answer from the model response."""
    # Find all **FINAL ANSWER: xxx** patterns and take the LAST one (skip examples in prompt)
    final_answer_matches = re.findall(r'\*\*FINAL ANSWER:\s*(.+?)\*\*', response, re.IGNORECASE)
    if final_answer_matches:
        # Take the last match (actual answer, not examples)
        answer = final_answer_matches[-1].strip()
        # Skip if it's the example placeholder
        if answer != '[your answer]':
            answer = answer.rstrip('.')
            answer = answer.strip('*')
            return answer

    # Try other patterns
    patterns = [
        r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
        r'final answer[:\s]+(.+?)(?:\n|$)',
        r'the answer is[:\s]+(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            # Take the last match
            answer = matches[-1].strip()
            if answer and answer != '[your answer]':
                answer = answer.rstrip('.')
                answer = answer.strip('*')
                return answer

    # Try to find boxed answer in LaTeX (last occurrence)
    boxed_matches = re.findall(r'\\boxed\{(.+?)\}', response)
    if boxed_matches:
        return boxed_matches[-1]

    return None


def clean_answer(answer: str) -> str:
    """Clean up extracted answer by removing LaTeX delimiters."""
    if answer is None:
        return None

    # Remove LaTeX display math delimiters
    answer = answer.strip()

    # Remove \boxed{} wrapper (can be nested or have content)
    while '\\boxed{' in answer:
        match = re.search(r'\\boxed\{(.+)\}', answer)
        if match:
            answer = match.group(1)
        else:
            break

    answer = re.sub(r'^\\\((.+)\\\)$', r'\1', answer)  # \( ... \)
    answer = re.sub(r'^\$(.+)\$$', r'\1', answer)      # $ ... $
    answer = re.sub(r'^\\\[(.+)\\\]$', r'\1', answer)  # \[ ... \]
    answer = re.sub(r'^\$\$(.+)\$\$$', r'\1', answer)  # $$ ... $$

    return answer.strip()


def solve_problem(agent, problem_data: dict, verbose: bool = True) -> dict:
    """Solve a single problem using the agent."""
    problem = problem_data['problem']
    answer = problem_data['answer']
    subject = problem_data.get('subject', None)
    level = problem_data.get('level', None)
    unique_id = problem_data.get('unique_id', 'unknown')

    prompt = create_prompt(problem, subject, level)

    result = {
        'id': unique_id,
        'subject': subject,
        'level': level,
        'problem': problem,
        'expected_answer': answer,
        'model_response': None,
        'extracted_answer': None,
        'is_correct': False,
        'error': None,
        'steps': []
    }

    try:
        # Run the agent
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
                msg_type = type(msg).__name__

                if verbose:
                    if msg_type == 'AIMessage':
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                result['steps'].append({
                                    'type': 'tool_call',
                                    'tool': tc.get('name', 'unknown'),
                                    'args': tc.get('args', {})
                                })
                    elif msg_type == 'ToolMessage':
                        result['steps'].append({
                            'type': 'tool_result',
                            'content': str(content)[:500]  # Truncate long outputs
                        })

                if isinstance(content, str):
                    full_response += content + "\n"

        result['model_response'] = full_response

        # Extract final answer
        extracted = extract_answer(full_response)
        extracted = clean_answer(extracted)
        result['extracted_answer'] = extracted

        # Check answer using MATH-500's check_answer
        if extracted:
            result['is_correct'] = check_answer(extracted, answer, timeout=False)
        else:
            result['is_correct'] = False

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Error: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description='MATH-500 Problem Solver')
    parser.add_argument('--model', type=str, default='gpt',
                        choices=['gpt', 'gemini', 'claude', 'kimi'],
                        help='Model to use for solving')
    parser.add_argument('--num_problems', type=int, default=100,
                        help='Number of problems to solve (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Model temperature')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed solving process')
    parser.add_argument('--subject', type=str, default=None,
                        help='Filter by subject (e.g., Algebra, Geometry)')

    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'environment', 'MATH-500', 'dataset', 'test.jsonl')
    output_dir = os.path.join(base_dir, 'my_code', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"\n{'='*60}")
    print(f"MATH-500 Problem Solver")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")

    problems = load_dataset(dataset_path, args.num_problems)

    # Filter by subject if specified
    if args.subject:
        problems = [p for p in problems if p.get('subject', '').lower() == args.subject.lower()]
        print(f"Filtered by subject: {args.subject}")

    print(f"Problems to solve: {len(problems)}")
    print(f"{'='*60}\n")

    # Create agent with tools
    agent = get_agent(args.model, tools=math_tools, temperature=args.temperature)

    # Solve problems
    results = []
    correct_count = 0
    subject_stats = {}

    pbar = tqdm(problems, desc=f"Solving with {args.model}", unit="problem")

    for problem_data in pbar:
        problem_id = problem_data.get('unique_id', len(results))
        subject = problem_data.get('subject', 'Unknown')

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Problem {problem_id} [{subject}]:")
            print(f"{problem_data['problem'][:200]}...")
            print(f"{'='*60}")

        result = solve_problem(agent, problem_data, verbose=args.verbose)
        results.append(result)

        if result['is_correct']:
            correct_count += 1

        # Track subject statistics
        if subject not in subject_stats:
            subject_stats[subject] = {'total': 0, 'correct': 0}
        subject_stats[subject]['total'] += 1
        if result['is_correct']:
            subject_stats[subject]['correct'] += 1

        # Update progress bar
        accuracy = correct_count / len(results) * 100
        pbar.set_postfix({
            'correct': correct_count,
            'accuracy': f'{accuracy:.1f}%'
        })

        if args.verbose:
            print(f"\nExpected: {result['expected_answer']}")
            print(f"Extracted: {result['extracted_answer']}")
            print(f"Correct: {'Yes' if result['is_correct'] else 'No'}")
            if result['steps']:
                print(f"\nTool calls: {len([s for s in result['steps'] if s['type'] == 'tool_call'])}")

    # Calculate final statistics
    accuracy = correct_count / len(results) * 100

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Total Problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nBy Subject:")
    print(f"{'-'*40}")
    for subject, stats in sorted(subject_stats.items()):
        subj_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {subject}: {stats['correct']}/{stats['total']} ({subj_acc:.1f}%)")
    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'math500_{args.model}_{timestamp}.json')

    output_data = {
        'model': args.model,
        'temperature': args.temperature,
        'num_problems': len(results),
        'correct': correct_count,
        'accuracy': accuracy,
        'subject_stats': subject_stats,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
