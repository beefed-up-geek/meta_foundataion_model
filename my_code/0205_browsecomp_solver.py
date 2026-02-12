"""
BrowseComp-Plus Problem Solver using LangGraph ReAct Agent
Supports multi-hop tool calling with GPT, Gemini, Claude, Kimi models.
Uses FAISS search + OpenRouter embeddings for document retrieval.

Usage:
    python 2026-02-05_browsecomp_solver.py --model gpt
    python 2026-02-05_browsecomp_solver.py --model claude --num_problems 100
"""

import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import json
import argparse
import importlib.util
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.Agents import get_agent, get_llm, AVAILABLE_MODELS

# Import from BrowseComp-Plus (handle hyphen in folder name)
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools_module = import_from_path("tools", os.path.join(base_dir, "environment", "BrowseComp-Plus", "agent_tools", "tools.py"))
check_module = import_from_path("check_answer", os.path.join(base_dir, "environment", "BrowseComp-Plus", "check_answer.py"))

# Tools: search, get_document
browse_tools = tools_module.ALL_TOOLS
check_answer = check_module.check_answer


def load_dataset(dataset_path: str, num_problems: int = None, skip: int = 0) -> list:
    """Load BrowseComp-Plus dataset from JSONL file."""
    problems = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    # Skip first N problems
    if skip > 0:
        problems = problems[skip:]

    if num_problems and num_problems < len(problems):
        problems = problems[:num_problems]

    return problems


def create_prompt(query: str) -> str:
    """Create a prompt for the agent to answer the research question."""
    return f"""You are an expert research assistant solving complex research questions using a document search system.

RESEARCH QUESTION:
{query}

AVAILABLE TOOLS:
1. search(query: str) - Search the knowledge base. Returns top-5 documents with docid, score, and snippet.
2. get_document(docid: str) - Retrieve the full text of a document by its ID.

RESEARCH STRATEGY:
1. Carefully analyze the question to identify KEY ENTITIES: names, places, dates, years, events, organizations.
2. Start by searching for the most specific and unique terms from the question.
3. Try MULTIPLE search queries (at least 3-5) using different combinations of keywords.
4. When a snippet looks promising, ALWAYS use get_document to read the full text.
5. Cross-reference information: if the question asks about X related to Y, search for both X and Y.
6. Pay attention to specific constraints like years, numbers, and precise facts.

SEARCH TIPS:
- Search for quoted phrases from the question
- Search for specific years/dates mentioned
- Search for proper nouns (people, places, organizations)
- If one search fails, try synonyms or related terms
- Combine multiple keywords to narrow results

CRITICAL:
- The answer EXISTS in the corpus. Keep trying different searches.
- You MUST provide a specific answer. "Unable to determine" is NEVER acceptable.
- When uncertain, provide your BEST guess based on evidence found.
- Read full documents using get_document when snippets seem relevant.

FINAL ANSWER FORMAT (required):
**FINAL ANSWER: [your specific answer here]**

The answer should be concise: a name, date, number, place, or specific fact.

Begin your research:"""


def extract_answer(response: str) -> str:
    """Extract the final answer from the model response."""
    # Find all **FINAL ANSWER: xxx** patterns and take the LAST one
    final_answer_matches = re.findall(r'\*\*FINAL ANSWER:\s*(.+?)\*\*', response, re.IGNORECASE)
    if final_answer_matches:
        answer = final_answer_matches[-1].strip()
        if answer != '[your precise answer]':
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
            answer = matches[-1].strip()
            if answer and answer != '[your precise answer]':
                answer = answer.rstrip('.')
                answer = answer.strip('*')
                return answer

    return None


def solve_problem(agent, problem_data: dict, verbose: bool = True) -> dict:
    """Solve a single research problem using the agent."""
    query = problem_data['query']
    answer = problem_data['answer']
    query_id = problem_data.get('query_id', 'unknown')

    prompt = create_prompt(query)

    result = {
        'query_id': query_id,
        'query': query,
        'expected_answer': answer,
        'model_response': None,
        'extracted_answer': None,
        'is_correct': False,
        'judge_result': None,
        'error': None,
        'steps': []
    }

    try:
        # Run the agent
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"recursion_limit": 100}
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
        result['extracted_answer'] = extracted

        # Check answer using LLM-as-Judge
        if extracted:
            judge_result = check_answer(
                question=query,
                response=full_response,
                correct_answer=answer,
                return_details=True
            )
            result['judge_result'] = judge_result
            result['is_correct'] = judge_result.get('correct', False)
        else:
            result['is_correct'] = False

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Error: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description='BrowseComp-Plus Problem Solver')
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
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N problems (default: 0)')

    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'environment', 'BrowseComp-Plus', 'dataset', 'browsecomp_plus_decrypted.jsonl')
    output_dir = os.path.join(base_dir, 'my_code', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download the dataset first using:")
        print("  cd environment/BrowseComp-Plus/dataset && python download.py")
        return

    # Load dataset
    print(f"\n{'='*60}")
    print(f"BrowseComp-Plus Problem Solver")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")

    problems = load_dataset(dataset_path, args.num_problems, skip=args.skip)
    print(f"Problems to solve: {len(problems)}")
    print(f"{'='*60}\n")

    # Create agent with tools
    print("Initializing search tools (loading FAISS index)...")
    try:
        agent = get_agent(args.model, tools=browse_tools, temperature=args.temperature)
        print("Tools initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing tools: {e}")
        print("\nMake sure you have:")
        print("1. Downloaded the dataset and indexes")
        print("2. Set OPENROUTER_API_KEY environment variable")
        return

    # Solve problems
    results = []
    correct_count = 0

    pbar = tqdm(problems, desc=f"Solving with {args.model}", unit="problem")

    for problem_data in pbar:
        query_id = problem_data.get('query_id', len(results))

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Query {query_id}:")
            print(f"{problem_data['query'][:200]}...")
            print(f"{'='*60}")

        result = solve_problem(agent, problem_data, verbose=args.verbose)
        results.append(result)

        if result['is_correct']:
            correct_count += 1

        # Update progress bar
        accuracy = correct_count / len(results) * 100
        pbar.set_postfix({
            'correct': correct_count,
            'accuracy': f'{accuracy:.1f}%'
        })

        if args.verbose:
            print(f"\nExpected: {result['expected_answer'][:100]}...")
            print(f"Extracted: {result['extracted_answer'][:100] if result['extracted_answer'] else 'None'}...")
            print(f"Correct: {'Yes' if result['is_correct'] else 'No'}")
            if result['steps']:
                search_calls = len([s for s in result['steps'] if s['type'] == 'tool_call' and s['tool'] == 'search'])
                doc_calls = len([s for s in result['steps'] if s['type'] == 'tool_call' and s['tool'] == 'get_document'])
                print(f"\nTool calls: search={search_calls}, get_document={doc_calls}")

    # Calculate final statistics
    accuracy = correct_count / len(results) * 100

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Total Problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'browsecomp_{args.model}_{timestamp}.json')

    output_data = {
        'model': args.model,
        'temperature': args.temperature,
        'num_problems': len(results),
        'correct': correct_count,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
