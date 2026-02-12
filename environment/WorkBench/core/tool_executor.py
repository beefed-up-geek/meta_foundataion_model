"""Tool Executor for WorkBench actions"""

import sys
import os
import re
from pathlib import Path
from typing import Tuple, Any


def _save_dataframes_to_csv(tool_module, module_name: str, workspace_path: Path):
    """Tool 모듈의 DataFrame을 CSV 파일로 저장

    원본 WorkBench tools는 global DataFrame을 메모리에서만 수정합니다.
    평가를 위해 변경된 DataFrame을 다시 CSV로 저장합니다.

    Note: workspace 파일명은 정답 디렉터리와 일치하도록 설정됨
    - project_tasks.csv → projects.csv
    - customer_relationship_manager_data.csv → customers.csv
    """
    csv_mapping = {
        "calendar": ("CALENDAR_EVENTS", "calendar_events.csv"),
        "email": ("EMAILS", "emails.csv"),
        "analytics": ("ANALYTICS_DATA", "analytics_data.csv"),
        # 정답 디렉터리와 일치하는 파일명 사용
        "project_management": ("PROJECT_TASKS", "projects.csv"),
        "customer_relationship_manager": ("CRM_DATA", "customers.csv"),
    }

    if module_name in csv_mapping:
        var_name, csv_name = csv_mapping[module_name]
        if hasattr(tool_module, var_name):
            df = getattr(tool_module, var_name)
            csv_path = workspace_path / csv_name
            df.to_csv(csv_path, index=False)


def execute_action(workspace_path: Path, action: str) -> Tuple[bool, Any]:
    """workspace context에서 action 실행

    Args:
        workspace_path: workspace 디렉토리 경로
        action: 실행할 명령어 (예: "analytics.create_plot.func(time_min='2023-11-21', ...)")

    Returns:
        (success, result): 성공 여부와 실행 결과
    """
    # 1. 원본 WorkBench tool 모듈 import를 위한 경로 설정
    # workspace_path를 절대 경로로 변환
    workspace_path = Path(workspace_path).absolute()

    # workspace_path: .../meta_foundataion_model/environment/WorkBench/_temp_workspace/20260204_xxxxx
    project_root = workspace_path.parent.parent.parent.parent  # meta_foundataion_model
    workbench_root = project_root / "env" / "WorkBench"
    workbench_src = workbench_root / "src"

    if not workbench_src.exists():
        return False, f"WorkBench source not found at {workbench_src}"

    # Add both paths: workbench_root for "from src.xxx" imports, workbench_src for "from tools.xxx" imports
    if str(workbench_root) not in sys.path:
        sys.path.insert(0, str(workbench_root))
    if str(workbench_src) not in sys.path:
        sys.path.insert(0, str(workbench_src))

    # 2. 현재 작업 디렉토리를 workspace로 변경 (CSV 파일들을 읽을 수 있도록)
    original_cwd = os.getcwd()
    original_data_path = None

    try:
        # workspace의 CSV 파일들을 "data/processed/" 경로처럼 보이게 설정
        # 원본 tool들이 "data/processed/calendar_events.csv" 형식으로 읽기 때문

        # 임시로 심볼릭 링크 생성
        temp_data_dir = workspace_path / "data"
        temp_data_dir.mkdir(exist_ok=True)
        (temp_data_dir / "processed").mkdir(exist_ok=True)
        (temp_data_dir / "raw").mkdir(exist_ok=True)

        # CSV 파일들을 링크 (workspace 파일명 → 원본 tools가 기대하는 파일명)
        # 파일명 매핑: workspace에서 사용하는 이름 → tools가 기대하는 이름
        file_name_mapping = {
            "projects.csv": "project_tasks.csv",
            "customers.csv": "customer_relationship_manager_data.csv",
        }

        for csv_file in workspace_path.glob("*.csv"):
            if csv_file.name == "email_addresses.csv":
                link_path = temp_data_dir / "raw" / csv_file.name
            else:
                # 매핑된 파일명이 있으면 사용, 없으면 원본 파일명 사용
                target_name = file_name_mapping.get(csv_file.name, csv_file.name)
                link_path = temp_data_dir / "processed" / target_name

            if not link_path.exists():
                link_path.symlink_to(csv_file.absolute())

        # 작업 디렉토리 변경
        os.chdir(workspace_path)

        # 3. Action 파싱 및 실행
        # 예: "analytics.create_plot.func(time_min='2023-11-21', ...)"
        match = re.match(r"(\w+)\.(\w+)\.func\((.*)\)", action)
        if not match:
            return False, f"Invalid action format: {action}"

        module_name, function_name, params_str = match.groups()

        # 4. 모듈 import
        try:
            if module_name == "calendar":
                from tools import calendar as tool_module
            elif module_name == "email":
                from tools import email as tool_module
            elif module_name == "analytics":
                from tools import analytics as tool_module
            elif module_name == "project_management":
                from tools import project_management as tool_module
            elif module_name == "customer_relationship_manager":
                from tools import customer_relationship_manager as tool_module
            elif module_name == "company_directory":
                from tools import company_directory as tool_module
            else:
                return False, f"Unknown module: {module_name}"

        except ImportError as e:
            return False, f"Failed to import module {module_name}: {e}"

        # 5. 상태 초기화 (매 실행마다 CSV를 새로 로드)
        # 원본 tools는 전역 DataFrame을 사용하므로, 각 쿼리마다 reset 필요
        if hasattr(tool_module, 'reset_state'):
            tool_module.reset_state()

        # 6. 함수 가져오기
        if not hasattr(tool_module, function_name):
            return False, f"Function {function_name} not found in module {module_name}"

        tool = getattr(tool_module, function_name)

        # LangChain tool에서 실제 함수 가져오기
        if hasattr(tool, 'func'):
            func = tool.func
        else:
            func = tool

        # 6. 파라미터 파싱 및 함수 호출
        # 간단한 eval 사용 (프로덕션에서는 더 안전한 파싱 필요)
        try:
            result = eval(f"func({params_str})")

            # 7. 변경된 DataFrame을 CSV로 저장 (원본 tools는 메모리만 수정함)
            _save_dataframes_to_csv(tool_module, module_name, workspace_path)

            # 8. analytics.create_plot 결과를 user_analytics.csv에 저장
            # 정답 평가는 이 파일에서 plot 경로를 확인함
            if module_name == "analytics" and function_name == "create_plot":
                import pandas as pd
                result_df = pd.DataFrame({"file_path": [result]})
                result_df.to_csv(workspace_path / "user_analytics.csv", index=False)

            return True, result
        except Exception as e:
            return False, f"Execution error: {e}"

    except Exception as e:
        return False, f"Unexpected error: {e}"

    finally:
        # 7. 원래 디렉토리로 복귀 및 정리
        os.chdir(original_cwd)

        # 심볼릭 링크 정리
        temp_data_dir = workspace_path / "data"
        if temp_data_dir.exists():
            import shutil
            shutil.rmtree(temp_data_dir)
