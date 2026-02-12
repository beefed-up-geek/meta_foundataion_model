# BrowseComp-Plus Environment

Deep-Research LLM Agent 평가를 위한 벤치마크 환경.

- Paper: https://arxiv.org/abs/2508.06600
- Code: https://github.com/texttron/BrowseComp-Plus

## Setup

### 1. 데이터셋 다운로드

```bash
cd dataset
python download.py
```

### 2. 환경 변수 설정

`.env` 파일에 OpenRouter API 키 추가:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. 의존성 설치

```bash
pip install faiss-cpu numpy openai langchain-core datasets huggingface_hub
```

## Usage

```python
from dotenv import load_dotenv
load_dotenv()

from tool import search, get_document

# 검색 (top-5)
results = search.invoke({"query": "Queen Arwa University 2002"})

# 문서 조회
doc = get_document.invoke({"docid": "5412"})
```

## Tool Specifications

| Tool | Description |
|------|-------------|
| `search` | Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. |
| `get_document` | Retrieve a full document by its docid. |

## Structure

```
environment/BrowseComp-Plus/
├── dataset/
│   ├── download.py                    # 데이터 다운로드 스크립트
│   ├── corpus.jsonl                   # 100K 문서 (다운로드 필요)
│   ├── browsecomp_plus_decrypted.jsonl # 830 쿼리 (다운로드 필요)
│   └── indexes/qwen3-embedding-8b/    # FAISS 인덱스 (다운로드 필요)
├── tool/
│   ├── __init__.py
│   └── tools.py                       # LangChain 도구 (FAISS + OpenRouter)
└── topics-qrels/
    ├── queries.tsv                    # 쿼리 목록
    ├── qrel_evidence.txt              # Evidence relevance judgments
    └── qrel_golds.txt                 # Gold relevance judgments
```
