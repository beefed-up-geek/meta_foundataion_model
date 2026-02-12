"""
WorkBench Tool-Augmented Solver - ReAct Agent with Workspace Tools
Uses models.py for LLM instances with LangGraph ReAct Agent.

Usage:
    python 0211_workbench_tool.py --model gemini-3-flash --num_problems 10 --benchmark calendar
    python 0211_workbench_tool.py --model claude-sonnet --reasoning --num_problems 30 --benchmark multi_domain
"""

import os
import sys
import json
import shutil
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Any, Callable
from functools import partial
from tqdm import tqdm

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKBENCH_DIR = os.path.join(BASE_DIR, "environment", "WorkBench")
TEMP_WORKSPACE_DIR = os.path.join(WORKBENCH_DIR, "_temp_workspace")

# Add paths for imports
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, WORKBENCH_DIR)

from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import create_react_agent
from environment.models import MODELS, MODELS_REASONING

# Import from WorkBench as a package
from core.workspace_manager import WorkspaceManager
from core import tools as core_tools_module
from check_answer import check_answer
from agent_tools import tools as agent_tools_module

# Available benchmarks
BENCHMARKS = [
    "calendar",
    "email",
    "analytics",
    "project_management",
    "customer_relationship_manager",
    "multi_domain"
]


class WorkspaceTracker:
    """Tracks all workspace directories created during a problem solving session."""

    def __init__(self, workspace_manager: WorkspaceManager):
        self.workspace_manager = workspace_manager
        self.created_workspaces: List[str] = []
        self.current_workspace_id: str = None

    def create_initial_workspace(self, query: str = "") -> str:
        """Create the initial workspace and track it."""
        workspace_id = self.workspace_manager.create_workspace(query)
        self.created_workspaces.append(workspace_id)
        self.current_workspace_id = workspace_id
        return workspace_id

    def track_workspace(self, workspace_id: str):
        """Track a newly created workspace."""
        if workspace_id not in self.created_workspaces:
            self.created_workspaces.append(workspace_id)
        self.current_workspace_id = workspace_id

    def cleanup_all(self):
        """Delete all tracked workspaces."""
        for ws_id in self.created_workspaces:
            try:
                self.workspace_manager.delete_workspace(ws_id)
            except Exception as e:
                print(f"Warning: Failed to delete workspace {ws_id}: {e}")
        self.created_workspaces.clear()
        self.current_workspace_id = None

    def get_final_workspace_path(self) -> Path:
        """Get the path to the final workspace (for answer checking)."""
        if self.current_workspace_id:
            return self.workspace_manager.get_workspace_path(self.current_workspace_id)
        return None


def create_connected_tools(tracker: WorkspaceTracker) -> List[StructuredTool]:
    """Create tools that are connected to the workspace manager.

    Each tool execution will:
    1. Fork the current workspace
    2. Execute the action
    3. Track the new workspace
    4. Return the result
    """
    connected_tools = []

    # Tool name to core function mapping
    tool_functions = {
        # Calendar tools
        "calendar_search_events": core_tools_module.calendar_search_events,
        "calendar_create_event": core_tools_module.calendar_create_event,
        "calendar_get_event_information_by_id": core_tools_module.calendar_get_event_information_by_id,
        "calendar_update_event": core_tools_module.calendar_update_event,
        "calendar_delete_event": core_tools_module.calendar_delete_event,

        # Email tools
        "email_search_emails": core_tools_module.email_search_emails,
        "email_send_email": core_tools_module.email_send_email,
        "email_get_email_information_by_id": core_tools_module.email_get_email_information_by_id,
        "email_forward_email": core_tools_module.email_forward_email,
        "email_reply_email": core_tools_module.email_reply_email,
        "email_delete_email": core_tools_module.email_delete_email,

        # Analytics tools
        "analytics_create_plot": core_tools_module.analytics_create_plot,
        "analytics_total_visits_count": core_tools_module.analytics_total_visits_count,
        "analytics_engaged_users_count": core_tools_module.analytics_engaged_users_count,
        "analytics_traffic_source_count": core_tools_module.analytics_traffic_source_count,
        "analytics_get_average_session_duration": core_tools_module.analytics_get_average_session_duration,
        "analytics_get_visitor_information_by_id": core_tools_module.analytics_get_visitor_information_by_id,

        # Project Management tools
        "project_management_search_tasks": core_tools_module.project_management_search_tasks,
        "project_management_create_task": core_tools_module.project_management_create_task,
        "project_management_get_task_information_by_id": core_tools_module.project_management_get_task_information_by_id,
        "project_management_update_task": core_tools_module.project_management_update_task,
        "project_management_delete_task": core_tools_module.project_management_delete_task,

        # CRM tools
        "customer_relationship_manager_search_customers": core_tools_module.customer_relationship_manager_search_customers,
        "customer_relationship_manager_add_customer": core_tools_module.customer_relationship_manager_add_customer,
        "customer_relationship_manager_update_customer": core_tools_module.customer_relationship_manager_update_customer,
        "customer_relationship_manager_delete_customer": core_tools_module.customer_relationship_manager_delete_customer,

        # Company Directory tools
        "company_directory_find_email_address": core_tools_module.company_directory_find_email_address,
    }

    # Create wrapper functions for each tool
    for agent_tool in agent_tools_module.ALL_TOOLS:
        tool_name = agent_tool.name
        core_func = tool_functions.get(tool_name)

        if core_func is None:
            print(f"Warning: No core function found for {tool_name}")
            continue

        # Get the original tool's schema
        original_schema = agent_tool.args_schema

        # Create a wrapper that uses the tracker
        def make_wrapper(func, name):
            def wrapper(**kwargs):
                # Get current workspace ID
                workspace_id = tracker.current_workspace_id
                if workspace_id is None:
                    return "Error: No active workspace"

                try:
                    # Call the core function with workspace_id
                    new_workspace_id, success, result = func(workspace_id, **kwargs)

                    # Track the new workspace
                    tracker.track_workspace(new_workspace_id)

                    if success:
                        return str(result)
                    else:
                        return f"Error: {result}"
                except Exception as e:
                    return f"Error executing {name}: {str(e)}"
            return wrapper

        wrapped_func = make_wrapper(core_func, tool_name)

        # Create the connected tool
        connected_tool = StructuredTool.from_function(
            func=wrapped_func,
            name=tool_name,
            description=agent_tool.description,
            args_schema=original_schema,
        )
        connected_tools.append(connected_tool)

    return connected_tools


def load_dataset(benchmark: str, num_problems: int = None, skip: int = 0) -> list:
    """Load WorkBench dataset from JSON file."""
    dataset_path = os.path.join(WORKBENCH_DIR, "dataset", "queries", f"{benchmark}.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    # Skip first N problems
    if skip > 0:
        problems = problems[skip:]

    if num_problems and num_problems < len(problems):
        problems = problems[:num_problems]

    return problems


SYSTEM_PROMPT = """You are an intelligent office assistant with access to various workplace tools.
You can manage calendars, emails, analytics, projects, customer relationships, and company directory.

TODAY'S DATE: 2023-11-30 (Thursday)

AVAILABLE TOOLS:
1. Calendar Tools: Search, create, update, delete calendar events
2. Email Tools: Search, send, reply, forward, delete emails
3. Analytics Tools: Get visit counts, engaged users, traffic sources, create plots
4. Project Management Tools: Search, create, update, delete tasks
5. CRM Tools: Search, add, update, delete customers
6. Company Directory: Find email addresses by name

IMPORTANT GUIDELINES:
1. Carefully analyze the user's request to understand what they need
2. Use the appropriate tools to complete the task
3. When searching, try different queries if the first one doesn't find what you need
4. For time-related queries, remember today is 2023-11-30
5. When creating events/tasks, use the correct date format: "YYYY-MM-DD HH:MM:SS" for datetime, "YYYY-MM-DD" for date
6. Duration is in minutes (e.g., "30" for 30 minutes, "60" for 1 hour)
7. Complete all necessary actions to fulfill the user's request
8. If a task requires multiple steps, complete them in order

When you have completed the task, summarize what you did."""


def create_prompt(query: str) -> str:
    """Create a prompt for the agent to answer the query."""
    return f"{SYSTEM_PROMPT}\n\nUSER REQUEST:\n{query}\n\nPlease complete this request using the available tools."


def solve_problem(
    agent,
    problem_data: dict,
    workspace_manager: WorkspaceManager,
    answer_base_dir: str,
    verbose: bool = False
) -> dict:
    """Solve a single problem using the agent."""
    query = problem_data['query']
    query_id = problem_data.get('query_id', 'unknown')
    ground_truth_result = problem_data.get('ground_truth_result', '')

    # Create workspace tracker for this problem
    tracker = WorkspaceTracker(workspace_manager)

    result = {
        'query_id': query_id,
        'query': query,
        'ground_truth_result': ground_truth_result,
        'model_response': None,
        'is_correct': False,
        'accuracy': 0.0,
        'check_details': None,
        'tool_calls': [],
        'workspaces_created': 0,
        'error': None
    }

    try:
        # Initialize workspace manager's tools
        core_tools_module.initialize_tools(workspace_manager)

        # Create initial workspace
        tracker.create_initial_workspace(query)

        # Create connected tools with the tracker
        connected_tools = create_connected_tools(tracker)

        # Create agent with connected tools
        problem_agent = create_react_agent(agent.model if hasattr(agent, 'model') else agent, tools=connected_tools)

        prompt = create_prompt(query)

        # Run the agent
        response = problem_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"recursion_limit": 50}
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
        result['workspaces_created'] = len(tracker.created_workspaces)

        # Check answer using check_answer.py
        final_workspace_path = tracker.get_final_workspace_path()

        if final_workspace_path and ground_truth_result:
            answer_dir = os.path.join(WORKBENCH_DIR, "dataset", ground_truth_result)

            if os.path.exists(answer_dir):
                try:
                    check_result = check_answer(str(final_workspace_path), answer_dir)
                    result['is_correct'] = check_result['accuracy'] == 1.0
                    result['accuracy'] = check_result['accuracy']
                    result['check_details'] = {
                        'total': check_result['total'],
                        'matched': check_result['matched'],
                        'results': {k: (v[0], v[1]) for k, v in check_result['results'].items()}
                    }
                except Exception as e:
                    result['error'] = f"Answer check error: {str(e)}"
            else:
                result['error'] = f"Answer directory not found: {answer_dir}"

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            import traceback
            traceback.print_exc()

    finally:
        # Clean up all workspaces created for this problem
        tracker.cleanup_all()

    return result


def main():
    parser = argparse.ArgumentParser(description='WorkBench Problem Solver (ReAct Agent with Tools)')
    parser.add_argument('--model', type=str, default='gemini-3-flash',
                        choices=list(MODELS.keys()),
                        help='Model to use for solving')
    parser.add_argument('--reasoning', action='store_true',
                        help='Use reasoning/thinking enabled model')
    parser.add_argument('--num_problems', type=int, default=10,
                        help='Number of problems to solve (default: 10)')
    parser.add_argument('--benchmark', type=str, default='calendar',
                        choices=BENCHMARKS,
                        help='Benchmark problem type (default: calendar)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N problems (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed solving process')

    args = parser.parse_args()

    # Setup paths
    output_dir = os.path.join(BASE_DIR, 'my_code', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create temp workspace directory if not exists
    os.makedirs(TEMP_WORKSPACE_DIR, exist_ok=True)

    # Select model
    model_dict = MODELS_REASONING if args.reasoning else MODELS
    llm = model_dict[args.model]
    model_suffix = "_reasoning" if args.reasoning else ""

    # Create workspace manager
    # Data files are in env/WorkBench, code is in environment/WorkBench
    env_workbench_dir = os.path.join(BASE_DIR, "env", "WorkBench")
    workspace_manager = WorkspaceManager(
        base_path=TEMP_WORKSPACE_DIR,
        template_base_path=env_workbench_dir
    )

    # Load dataset
    problems = load_dataset(args.benchmark, args.num_problems, skip=args.skip)

    print(f"\n{'='*60}")
    print(f"WorkBench Problem Solver (LangGraph ReAct Agent)")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Problems to solve: {len(problems)}")
    print(f"{'='*60}\n")

    # Solve problems
    results = []
    correct_count = 0
    total_tool_calls = 0
    total_workspaces = 0

    pbar = tqdm(problems, desc=f"Solving {args.benchmark}", unit="problem")

    for problem_data in pbar:
        result = solve_problem(
            llm,
            problem_data,
            workspace_manager,
            os.path.join(WORKBENCH_DIR, "dataset"),
            verbose=args.verbose
        )
        results.append(result)

        if result['is_correct']:
            correct_count += 1

        total_tool_calls += len(result.get('tool_calls', []))
        total_workspaces += result.get('workspaces_created', 0)

        accuracy = correct_count / len(results) * 100
        pbar.set_postfix({
            'correct': correct_count,
            'accuracy': f'{accuracy:.1f}%',
            'tools': total_tool_calls
        })

    # Calculate final statistics
    accuracy = correct_count / len(results) * 100
    avg_tool_calls = total_tool_calls / len(results) if results else 0
    avg_workspaces = total_workspaces / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}{model_suffix}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Total Problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Tool Calls: {total_tool_calls}")
    print(f"Avg Tool Calls per Problem: {avg_tool_calls:.2f}")
    print(f"Total Workspaces Created: {total_workspaces}")
    print(f"Avg Workspaces per Problem: {avg_workspaces:.2f}")
    print(f"{'='*60}")

    # Save results with benchmark name in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace('-', '_')
    benchmark_name = args.benchmark.replace('-', '_')
    output_file = os.path.join(output_dir, f'workbench_{benchmark_name}_{model_name}{model_suffix}_{timestamp}.json')

    output_data = {
        'model': args.model,
        'reasoning': args.reasoning,
        'benchmark': args.benchmark,
        'num_problems': len(results),
        'correct': correct_count,
        'accuracy': accuracy,
        'total_tool_calls': total_tool_calls,
        'avg_tool_calls': avg_tool_calls,
        'total_workspaces': total_workspaces,
        'avg_workspaces': avg_workspaces,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_file}")

    # Final cleanup: ensure temp workspace is clean
    try:
        remaining = list(Path(TEMP_WORKSPACE_DIR).iterdir())
        if remaining:
            print(f"\nCleaning up {len(remaining)} remaining workspace(s)...")
            for item in remaining:
                if item.is_dir():
                    shutil.rmtree(item)
            print("Cleanup complete.")
    except Exception as e:
        print(f"Warning: Cleanup error: {e}")


if __name__ == "__main__":
    main()
