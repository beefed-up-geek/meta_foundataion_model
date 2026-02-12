"""Simple Workspace Manager for file-system based state management"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Any


class WorkspaceManager:
    """파일 시스템 기반 workspace 관리 - 단순 버전

    핵심 기능:
    1. create_workspace() - 기본 CSV로 워크스페이스 생성
    2. delete_workspace() - 워크스페이스 삭제
    3. fork_and_execute() - 워크스페이스 포크 & 명령어 실행
    """

    def __init__(self, base_path: str = "_temp_workspace", template_base_path: str = None):
        """WorkspaceManager 초기화

        Args:
            base_path: workspace 저장 디렉토리
            template_base_path: 원본 CSV 경로 (None이면 자동 탐지)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # 원본 CSV 경로 설정
        if template_base_path:
            self.template_base = Path(template_base_path)
        else:
            project_root = Path(__file__).parent.parent.parent.parent
            self.template_base = project_root / "env" / "WorkBench"

        # CSV 파일 목록 (정답 디렉터리와 일치하도록 파일명 매핑)
        # key = workspace에서 사용할 파일명, value = 원본 파일 경로
        self.csv_files = {
            "calendar_events.csv": self.template_base / "data" / "processed" / "calendar_events.csv",
            "emails.csv": self.template_base / "data" / "processed" / "emails.csv",
            "analytics_data.csv": self.template_base / "data" / "processed" / "analytics_data.csv",
            # 정답 디렉터리는 projects.csv, customers.csv 사용
            "projects.csv": self.template_base / "data" / "processed" / "project_tasks.csv",
            "customers.csv": self.template_base / "data" / "processed" / "customer_relationship_manager_data.csv",
            "email_addresses.csv": self.template_base / "data" / "raw" / "email_addresses.csv",
        }

    def _generate_id(self) -> str:
        """날짜_해시 형태의 workspace ID 생성"""
        date_str = datetime.now().strftime("%Y%m%d")
        timestamp = datetime.now().isoformat()
        hash_str = hashlib.md5(timestamp.encode()).hexdigest()[:6]
        return f"{date_str}_{hash_str}"

    def create_workspace(self, query: str = "") -> str:
        """기본 CSV 파일들로 새 workspace 생성

        Args:
            query: 쿼리 문자열 (metadata에 기록)

        Returns:
            workspace_id: 생성된 workspace ID
        """
        workspace_id = self._generate_id()
        workspace_path = self.base_path / workspace_id
        workspace_path.mkdir(exist_ok=True)

        # CSV 파일들 복사
        for csv_name, source in self.csv_files.items():
            if not source.exists():
                raise FileNotFoundError(f"Template CSV not found: {source}")
            shutil.copy2(source, workspace_path / csv_name)

        # metadata.json 생성
        metadata = {
            "workspace_id": workspace_id,
            "parent_id": None,
            "created_at": datetime.now().isoformat(),
            "query": query,
            "actions": []
        }

        with open(workspace_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return workspace_id

    def delete_workspace(self, workspace_id: str):
        """workspace 삭제

        Args:
            workspace_id: 삭제할 workspace ID
        """
        workspace_path = self.base_path / workspace_id
        if workspace_path.exists():
            shutil.rmtree(workspace_path)

    def fork_and_execute(
        self,
        source_id: str,
        action: str,
        executor_func=None
    ) -> Tuple[str, bool, Any]:
        """workspace를 포크하고 명령어 실행

        Args:
            source_id: 원본 workspace ID
            action: 실행할 명령어 문자열
            executor_func: 명령어를 실행하는 함수 (workspace_path, action) -> (success, result)

        Returns:
            (new_workspace_id, success, result): 포크된 workspace ID, 성공 여부, 실행 결과
        """
        source_path = self.base_path / source_id
        if not source_path.exists():
            raise ValueError(f"Source workspace not found: {source_id}")

        # 새 workspace ID 생성 & 폴더 복제
        new_id = self._generate_id()
        new_path = self.base_path / new_id
        shutil.copytree(source_path, new_path)

        # metadata 업데이트
        metadata_path = new_path / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        metadata["workspace_id"] = new_id
        metadata["parent_id"] = source_id
        metadata["created_at"] = datetime.now().isoformat()

        # 명령어 실행
        success = False
        result = None

        if executor_func:
            try:
                success, result = executor_func(new_path, action)
            except Exception as e:
                success = False
                result = str(e)

        # 실행 결과를 metadata에 기록
        metadata["actions"].append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "result": str(result) if result is not None else None
        })

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return new_id, success, result

    def get_workspace_path(self, workspace_id: str) -> Path:
        """workspace 경로 반환"""
        return self.base_path / workspace_id
