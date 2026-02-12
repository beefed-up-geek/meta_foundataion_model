"""
BrowseComp-Plus Tool-Augmented Solver - ReAct Agent with Search Tools
Uses models.py for LLM instances with LangGraph ReAct Agent.

Prerequisites:
    1. Start FAISS server first:
       cd environment/BrowseComp-Plus/dataset && python faiss_server.py

Usage:
    python 0211_browsecomp_tool.py --model gemini-3-flash --num_problems 10
    python 0211_browsecomp_tool.py --model gemini-3-flash --reasoning --num_problems 30
    python 0211_browsecomp_tool.py --model gemini-3-flash --num_problems 30 --max_tool_calls 50
"""

import os
import sys
import re
import json
import argparse
import importlib.util
from datetime import datetime
from tqdm import tqdm

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.prebuilt import create_react_agent
from environment.models import MODELS, MODELS_REASONING

# Import from BrowseComp-Plus (handle hyphen in folder name)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import tools
spec_tools = importlib.util.spec_from_file_location(
    "tools",
    os.path.join(base_dir, "environment", "BrowseComp-Plus", "agent_tools", "tools.py")
)
tools_module = importlib.util.module_from_spec(spec_tools)
spec_tools.loader.exec_module(tools_module)
browse_tools = tools_module.ALL_TOOLS

# Import check_answer
spec_check = importlib.util.spec_from_file_location(
    "check_answer",
    os.path.join(base_dir, "environment", "BrowseComp-Plus", "check_answer.py")
)
check_module = importlib.util.module_from_spec(spec_check)
spec_check.loader.exec_module(check_module)
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


def get_system_prompt(max_tool_calls: int) -> str:
    """Generate system prompt with max tool calls limit."""
    return f"""You are an expert research assistant solving complex research questions using a document search system.

You have access to the following tools:
1. search(query: str) - Search the knowledge base. Returns top-5 documents with docid, score, and snippet.
2. get_document(docid: str) - Retrieve the full text of a document by its ID.

IMPORTANT CONSTRAINT:
- You have a MAXIMUM of {max_tool_calls} tool calls for this question.
- Use your tool calls efficiently and strategically.
- If you're running low on tool calls, prioritize getting to an answer.
- After approximately {max_tool_calls - 5} tool calls, you MUST provide your best answer.

RESEARCH STRATEGY:
1. Carefully analyze the question to identify KEY ENTITIES: names, places, dates, years, events, organizations.
2. Start by searching for the most specific and unique terms from the question.
3. Try MULTIPLE search queries (3-5) using different combinations of keywords.
4. When a snippet looks promising, use get_document to read the full text.
5. Cross-reference information: if the question asks about X related to Y, search for both X and Y.
6. Pay attention to specific constraints like years, numbers, and precise facts.

SEARCH TIPS:
- Search for quoted phrases from the question
- Search for specific years/dates mentioned
- Search for proper nouns (people, places, organizations)
- If one search fails, try synonyms or related terms
- Combine multiple keywords to narrow results

EFFICIENCY TIPS:
- Don't repeat the same search query
- If you found relevant documents, read them instead of searching more
- Synthesize information from multiple sources quickly
- Make a decision when you have reasonable evidence

CRITICAL:
- The answer EXISTS in the corpus. Keep trying different searches.
- You MUST provide a specific answer. "Unable to determine" is NEVER acceptable.
- When uncertain, provide your BEST guess based on evidence found.
- Read full documents using get_document when snippets seem relevant.

FINAL ANSWER FORMAT (required):
**FINAL ANSWER: [your specific answer here]**

The answer should be concise: a name, date, number, place, or specific fact."""


def create_prompt(query: str, max_tool_calls: int = 50) -> str:
    """Create a prompt for the agent to answer the research question."""
    system_prompt = get_system_prompt(max_tool_calls)
    return f"{system_prompt}\n\nRESEARCH QUESTION:\n{query}\n\nBegin your research:"


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


def solve_problem(agent, problem_data: dict, verbose: bool = False, max_tool_calls: int = 50) -> dict:
    """Solve a single research problem using the agent."""
    query = problem_data['query']
    answer = problem_data['answer']
    query_id = problem_data.get('query_id', 'unknown')

    prompt = create_prompt(query, max_tool_calls)

    result = {
        'query_id': query_id,
        'query': query,
        'expected_answer': answer,
        'model_response': None,
        'extracted_answer': None,
        'is_correct': False,
        'judge_result': None,
        'tool_calls': [],
        'error': None
    }

    try:
        # Run the agent with recursion limit based on max_tool_calls
        # Each tool call requires ~2 recursions (call + response)
        recursion_limit = max(max_tool_calls * 2 + 10, 50)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"recursion_limit": recursion_limit}
        )

        # Extract messages from response
        messages = response.get('messages', [])
        full_response = ""
        tool_calls = []

        for msg in messages:
            # Extract tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        'name': tc.get('name', 'unknown'),
                        'args': tc.get('args', {})
                    })

            # Extract content
            if hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, str):
                    full_response += content + "\n"
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            full_response += item['text'] + "\n"

        result['model_response'] = full_response
        result['tool_calls'] = tool_calls

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
    parser = argparse.ArgumentParser(description='BrowseComp-Plus Problem Solver (ReAct Agent with Tools)')
    parser.add_argument('--model', type=str, default='gemini-3-flash',
                        choices=list(MODELS.keys()),
                        help='Model to use for solving')
    parser.add_argument('--reasoning', action='store_true',
                        help='Use reasoning/thinking enabled model')
    parser.add_argument('--num_problems', type=int, default=10,
                        help='Number of problems to solve (default: 10)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N problems (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed solving process')
    parser.add_argument('--max_tool_calls', type=int, default=50,
                        help='Maximum tool calls per problem (default: 50)')

    args = parser.parse_args()

    # Setup paths
    dataset_path = os.path.join(base_dir, 'environment', 'BrowseComp-Plus', 'dataset', 'browsecomp_plus_decrypted.jsonl')
    output_dir = os.path.join(base_dir, 'my_code', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download the dataset first.")
        return

    # Select model
    model_dict = MODELS_REASONING if args.reasoning else MODELS
    llm = model_dict[args.model]
    model_suffix = "_reasoning" if args.reasoning else ""

    # Create agent with tools
    print(f"\n{'='*60}")
    print(f"BrowseComp-Plus Problem Solver (LangGraph ReAct Agent)")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")
    print(f"Tools: {[tool.name for tool in browse_tools]}")
    print(f"Max Tool Calls: {args.max_tool_calls}")

    # Check FAISS server
    import requests
    try:
        resp = requests.get("http://127.0.0.1:8765/health", timeout=5)
        if resp.status_code == 200:
            print("FAISS Server: Running")
        else:
            print("FAISS Server: Error")
            return
    except:
        print("FAISS Server: NOT RUNNING")
        print("\nPlease start the FAISS server first:")
        print("  cd environment/BrowseComp-Plus/dataset && python faiss_server.py")
        return

    # Create agent
    agent = create_react_agent(llm, tools=browse_tools)

    # Load dataset
    problems = load_dataset(dataset_path, args.num_problems, skip=args.skip)
    print(f"Problems to solve: {len(problems)}")
    print(f"{'='*60}\n")

    # Solve problems
    results = []
    correct_count = 0
    total_tool_calls = 0

    pbar = tqdm(problems, desc=f"Solving with {args.model}", unit="problem")

    for problem_data in pbar:
        result = solve_problem(agent, problem_data, verbose=args.verbose, max_tool_calls=args.max_tool_calls)
        results.append(result)

        if result['is_correct']:
            correct_count += 1

        total_tool_calls += len(result.get('tool_calls', []))

        accuracy = correct_count / len(results) * 100
        pbar.set_postfix({
            'correct': correct_count,
            'accuracy': f'{accuracy:.1f}%',
            'tools': total_tool_calls
        })

    # Calculate final statistics
    accuracy = correct_count / len(results) * 100
    avg_tool_calls = total_tool_calls / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")
    print(f"Tools: {[tool.name for tool in browse_tools]}")
    print(f"Max Tool Calls: {args.max_tool_calls}")
    print(f"Total Problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Tool Calls: {total_tool_calls}")
    print(f"Avg Tool Calls per Problem: {avg_tool_calls:.2f}")
    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace('-', '_')
    output_file = os.path.join(output_dir, f'browsecomp_{model_name}{model_suffix}_{timestamp}.json')

    output_data = {
        'model': args.model,
        'reasoning': args.reasoning,
        'tools': [tool.name for tool in browse_tools],
        'max_tool_calls': args.max_tool_calls,
        'num_problems': len(results),
        'correct': correct_count,
        'accuracy': accuracy,
        'total_tool_calls': total_tool_calls,
        'avg_tool_calls': avg_tool_calls,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
