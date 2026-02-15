#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MinerU OCR Skill CLI
~~~~~~~~~~~~~~~~~~~~

自包含脚本，不依赖 cn_annual_report_ocr。
支持单文件/目录/递归批量 OCR，包含多 Token 轮换与断点续跑。
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import urllib3
import fitz  # PyMuPDF

# Disable TLS warnings when SSL verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _get_skill_root() -> Path:
    try:
        current_dir = Path(__file__).parent.resolve()
    except NameError:
        # 交互式环境 __file__ 不存在时的回退策略
        current_dir = Path.cwd().resolve()
    return current_dir.parent


def _resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _collect_pdfs(base_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in base_dir.rglob("*.pdf") if p.is_file()])
    return sorted([p for p in base_dir.glob("*.pdf") if p.is_file()])


def _build_output_dir(out_root: Path, pdf_path: Path, base_dir: Path) -> Path:
    """Return output directory without creating a per-PDF subfolder."""
    try:
        rel = pdf_path.relative_to(base_dir)
        rel_parent = rel.parent
        return (out_root / rel_parent).resolve()
    except Exception:
        return out_root.resolve()


def _make_report_id(pdf_path: Path) -> str:
    path_hash = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:10]
    return f"{pdf_path.stem}_{path_hash}"


def _load_tokens_from_file(tokens_file: Path) -> List[str]:
    try:
        content = tokens_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"tokens 文件不存在：{tokens_file}")
    except Exception as e:
        raise RuntimeError(f"读取 tokens 文件失败：{tokens_file}；原因：{e}")

    tokens = []
    for line in content.splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        tokens.append(token)
    return tokens


def _check_token_available(token: str, api_base: str, test_url: str) -> Tuple[bool, str]:
    payload = {
        "files": [{"url": test_url, "data_id": "token_check"}],
        "model_version": "vlm",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{api_base}/extract/task/batch",
            headers=headers,
            json=payload,
            timeout=30,
        )
    except Exception as e:
        return False, f"请求失败: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"

    try:
        data = resp.json()
    except Exception as e:
        return False, f"解析响应失败: {e}"

    code = data.get("code")
    msg = data.get("msg", "")
    if code == 0:
        return True, "ok"
    if str(code) in {"A0202", "A0211"}:
        return False, f"{code} {msg}".strip()
    return False, f"{code} {msg}".strip()


class OCRStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class OCRResult:
    unique_id: str
    status: OCRStatus
    retry_count: int = 0
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    output_dir: Optional[Path] = None
    zip_path: Optional[Path] = None

    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "unique_id": self.unique_id,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "processing_time_seconds": self.processing_time_seconds,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "zip_path": str(self.zip_path) if self.zip_path else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Optional[str]]) -> "OCRResult":
        return cls(
            unique_id=data["unique_id"],
            status=OCRStatus(data["status"]),
            retry_count=int(data.get("retry_count", 0) or 0),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            error_message=data.get("error_message"),
            error_type=data.get("error_type"),
            processing_time_seconds=data.get("processing_time_seconds"),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            zip_path=Path(data["zip_path"]) if data.get("zip_path") else None,
        )


@dataclass
class Report:
    unique_id: str
    pdf_path: Path
    page_count: Optional[int] = None


class QuotaExhaustedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TokenQuotaManager:
    def __init__(self, tokens: List[str], daily_page_limit: int):
        if not tokens:
            raise ValueError("至少需要提供一个 MinerU API Token")
        self.tokens = tokens
        self.daily_page_limit = daily_page_limit
        self._lock = threading.Lock()
        self._current_index = 0
        self._usage = self._empty_usage()

    def _should_reset(self, last_reset: str) -> bool:
        try:
            last_dt = datetime.fromisoformat(last_reset)
        except (ValueError, TypeError):
            return True
        now = datetime.now()
        if last_dt.hour < 12:
            next_reset = last_dt.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            next_reset = last_dt.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return now >= next_reset

    def _empty_usage(self) -> Dict[str, Dict[str, int]]:
        return {
            "last_reset": datetime.now().isoformat(),
            "date": time.strftime("%Y-%m-%d"),
            "tokens": {t: {"pages": 0, "files": 0} for t in self.tokens},
        }

    def reserve_token(self, pages: int) -> str:
        with self._lock:
            start_index = self._current_index
            if self._should_reset(self._usage.get("last_reset", "")):
                self._usage = self._empty_usage()
            while True:
                token = self.tokens[self._current_index]
                token_usage = self._usage["tokens"].get(token, {"pages": 0, "files": 0})
                if token_usage["pages"] + pages <= self.daily_page_limit:
                    token_usage["pages"] += pages
                    token_usage["files"] += 1
                    self._usage["tokens"][token] = token_usage
                    return token
                self._current_index = (self._current_index + 1) % len(self.tokens)
                if self._current_index == start_index:
                    raise QuotaExhaustedError("所有 Token 均达到日解析页数上限")


class StateManager:
    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self._lock = threading.Lock()
        self._state: Dict[str, OCRResult] = {}
        self._dirty_keys: set = set()
        self._load_state()

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        unique_id = data.get("unique_id")
                        if unique_id:
                            self._state[unique_id] = OCRResult.from_dict(data)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return

    def _save_state(self, full_rewrite: bool = False):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if full_rewrite:
            with self.state_file.open("w", encoding="utf-8") as f:
                for result in self._state.values():
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        else:
            with self.state_file.open("a", encoding="utf-8") as f:
                for key in self._dirty_keys:
                    if key in self._state:
                        f.write(json.dumps(self._state[key].to_dict(), ensure_ascii=False) + "\n")
        self._dirty_keys.clear()

    def register_tasks(self, unique_ids: List[str]):
        with self._lock:
            for unique_id in unique_ids:
                if unique_id not in self._state:
                    self._state[unique_id] = OCRResult(
                        unique_id=unique_id,
                        status=OCRStatus.PENDING,
                    )
                    self._dirty_keys.add(unique_id)
            if self._dirty_keys:
                self._save_state(full_rewrite=False)

    def is_pending_or_failed(self, unique_id: str, max_retries: int = 3) -> bool:
        with self._lock:
            if unique_id not in self._state:
                return True
            result = self._state[unique_id]
            if result.status == OCRStatus.PENDING:
                return True
            if result.status == OCRStatus.FAILED and result.retry_count < max_retries:
                return True
            return False

    def get_result(self, unique_id: str) -> Optional[OCRResult]:
        with self._lock:
            return self._state.get(unique_id)

    def update_result(self, result: OCRResult, flush: bool = True):
        with self._lock:
            self._state[result.unique_id] = result
            self._dirty_keys.add(result.unique_id)
            if flush:
                self._save_state(full_rewrite=False)


class PDFValidator:
    def __init__(self, max_size_mb: int = 500, max_pages: int = 1000):
        self.max_size_mb = max_size_mb
        self.max_pages = max_pages

    def validate_pdf(self, pdf_path: Path) -> Tuple[bool, Optional[str], Optional[int]]:
        if not pdf_path.exists():
            return False, f"文件不存在: {pdf_path}", None
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            return False, f"文件过大: {file_size_mb:.1f}MB > {self.max_size_mb}MB", None
        try:
            doc = fitz.open(str(pdf_path))
            page_count = doc.page_count
            doc.close()
        except Exception as e:
            return False, f"PDF 无法读取: {e}", None
        if page_count > self.max_pages:
            return False, f"页数过多: {page_count} > {self.max_pages}", page_count
        if page_count == 0:
            return False, "PDF 页数为 0", 0
        return True, None, page_count


class MinerUAPIProvider:
    def __init__(self, token: str, api_base: str = "https://mineru.net/api/v4", timeout_seconds: int = 3600):
        self.token = token
        self.api_base = api_base
        self.timeout = timeout_seconds
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.trust_env = False
            self._session.verify = False
        return self._session

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def process_pdf(self, report: Report, output_dir: Path) -> OCRResult:
        start_time = time.time()
        try:
            batch_id = self._upload_pdf(report.pdf_path)
            download_url = self._poll_for_result(batch_id, report.unique_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            zip_path = output_dir / f"{report.pdf_path.stem}.zip"
            self._download_result(download_url, zip_path)
            extract_dir = output_dir / report.pdf_path.stem
            self._extract_zip(zip_path, extract_dir)
            try:
                zip_path.unlink()
            except Exception:
                pass
            self._cleanup_extract_dir(extract_dir)
            return OCRResult(
                unique_id=report.unique_id,
                status=OCRStatus.SUCCESS,
                processing_time_seconds=time.time() - start_time,
                output_dir=extract_dir,
                zip_path=None,
            )
        except Exception as e:
            return OCRResult(
                unique_id=report.unique_id,
                status=OCRStatus.FAILED,
                error_message=str(e),
                error_type=type(e).__name__,
                processing_time_seconds=time.time() - start_time,
            )

    def _upload_pdf(self, pdf_path: Path) -> str:
        file_name = pdf_path.name
        payload = {
            "files": [{"name": file_name, "data_id": f"file_{int(time.time())}"}],
            "model_version": "vlm",
        }
        session = self._get_session()
        resp = session.post(
            f"{self.api_base}/file-urls/batch",
            headers=self._headers(),
            json=payload,
            verify=False,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"申请上传链接失败: HTTP {resp.status_code}")
        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")
        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]
        with pdf_path.open("rb") as f:
            upload_resp = session.put(upload_url, data=f, verify=False, timeout=300)
        if upload_resp.status_code != 200:
            raise RuntimeError(f"文件上传失败: HTTP {upload_resp.status_code}")
        return batch_id

    def _poll_for_result(self, batch_id: str, unique_id: str = "") -> str:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"处理超时({self.timeout}s)")
            session = self._get_session()
            resp = session.get(
                f"{self.api_base}/extract-results/batch/{batch_id}",
                headers=self._headers(),
                verify=False,
                timeout=60,
            )
            if resp.status_code != 200:
                time.sleep(10)
                continue
            data = resp.json()
            if data.get("code") != 0:
                time.sleep(10)
                continue
            extract_results = data.get("data", {}).get("extract_result") or data.get("data", {}).get("extract_results")
            if not extract_results:
                time.sleep(10)
                continue
            result = extract_results[0]
            state = result.get("state")
            if state == "failed":
                raise RuntimeError(f"MinerU 处理失败: {result.get('err_msg', '未知错误')}")
            if state == "done":
                return result["full_zip_url"]
            if unique_id:
                print(f"[{unique_id}] 处理中... state={state} elapsed={elapsed:.0f}s")
            time.sleep(10)

    def _download_result(self, url: str, output_path: Path):
        session = self._get_session()
        resp = session.get(url, stream=True, verify=False, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"结果下载失败: HTTP {resp.status_code}")
        with output_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if output_path.stat().st_size < 1000:
            raise RuntimeError("下载文件过小，可能不完整")

    def _extract_zip(self, zip_path: Path, extract_dir: Path):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    def _cleanup_extract_dir(self, extract_dir: Path):
        """Keep only images/ and full.md in the extracted folder."""
        keep_files = {"full.md"}
        keep_dirs = {"images"}
        for p in extract_dir.iterdir():
            if p.is_dir():
                if p.name in keep_dirs:
                    continue
                shutil.rmtree(p, ignore_errors=True)
            else:
                if p.name in keep_files:
                    continue
                try:
                    p.unlink()
                except Exception:
                    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MinerU OCR Skill CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/ocr_mineru_cli.py --pdf "a.pdf" --out "data/ocr_out"
  python scripts/ocr_mineru_cli.py --dir "raw_data/pdf_2023" --out "data/ocr_out"
  python scripts/ocr_mineru_cli.py --dir "raw_data" --recursive --dry-run
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", type=str, help="单个 PDF 文件路径")
    group.add_argument("--dir", type=str, help="包含 PDF 的目录")

    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--out", type=str, default=None, help="输出目录（默认输入目录）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际调用 OCR")
    parser.add_argument("--limit", type=int, default=None, help="最多处理文件数量")
    parser.add_argument("--workers", type=int, default=1, help="并发线程数")
    parser.add_argument("--no-resume", action="store_true", help="不使用断点续跑")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="只生成并打印命令，不实际执行（供 Coding Agent 生成命令供用户手动运行）",
    )
    return parser.parse_args()


def _prepare_reports(
    pdf_paths: List[Path],
    base_dir: Path,
    out_root: Path,
) -> List[Tuple[Report, Path, str]]:
    reports = []
    for pdf_path in pdf_paths:
        unique_id = _make_report_id(pdf_path)
        report = Report(
            unique_id=unique_id,
            pdf_path=pdf_path,
        )
        output_dir = _build_output_dir(out_root, pdf_path, base_dir)
        reports.append((report, output_dir, unique_id))
    return reports


def _generate_command(args: argparse.Namespace, skill_root: Path, python_exe: str) -> str:
    """根据参数生成完整的 OCR 命令"""
    script_path = Path(__file__).resolve()
    python_path = python_exe if python_exe else "python"

    cmd_parts = [python_path, str(script_path)]

    if args.pdf:
        cmd_parts.append(f'--pdf "{args.pdf}"')
    if args.dir:
        cmd_parts.append(f'--dir "{args.dir}"')
    if args.recursive:
        cmd_parts.append("--recursive")
    if args.out:
        cmd_parts.append(f'--out "{args.out}"')
    if args.dry_run:
        cmd_parts.append("--dry-run")
    if args.limit:
        cmd_parts.append(f"--limit {args.limit}")
    if args.workers and args.workers > 1:
        cmd_parts.append(f"--workers {args.workers}")
    if args.no_resume:
        cmd_parts.append("--no-resume")

    return " ".join(cmd_parts)


def _process_single(
    report: Report,
    output_dir: Path,
    state_manager: StateManager,
    token_manager: TokenQuotaManager,
    pdf_validator: PDFValidator,
    api_base: str,
    timeout_seconds: int,
) -> OCRResult:
    is_valid, error_msg, page_count = pdf_validator.validate_pdf(report.pdf_path)
    if not is_valid:
        result = OCRResult(
            unique_id=report.unique_id,
            status=OCRStatus.FAILED,
            error_message=error_msg,
            error_type="FileValidationError",
        )
        state_manager.update_result(result)
        return result

    report.page_count = page_count
    token = token_manager.reserve_token(page_count or 0)

    processing_result = OCRResult(
        unique_id=report.unique_id,
        status=OCRStatus.PROCESSING,
    )
    state_manager.update_result(processing_result)

    provider = MinerUAPIProvider(token, api_base=api_base, timeout_seconds=timeout_seconds)
    result = provider.process_pdf(report, output_dir)

    if result.status != OCRStatus.SUCCESS:
        prev_result = state_manager.get_result(report.unique_id)
        if prev_result:
            result.retry_count = prev_result.retry_count + 1

    state_manager.update_result(result)
    return result


def main() -> int:
    args = _parse_args()
    skill_root = _get_skill_root()

    if args.pdf:
        pdf_path = _resolve_path(args.pdf)
        if not pdf_path.exists():
            print(f"文件不存在: {pdf_path}")
            return 1
        base_dir = pdf_path.parent
        pdf_paths = [pdf_path]
    else:
        base_dir = _resolve_path(args.dir)
        if not base_dir.exists():
            print(f"目录不存在: {base_dir}")
            return 1
        pdf_paths = _collect_pdfs(base_dir, args.recursive)

    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    out_root = _resolve_path(args.out) if args.out else base_dir
    state_file = skill_root / "state" / "ocr_skill_state.jsonl"

    print(f"发现 PDF 数量: {len(pdf_paths)}")
    if not pdf_paths:
        return 0

    # --generate-only 模式：只生成命令，不执行
    if args.generate_only:
        import sys

        # 检测 Python 解释器路径
        python_exe = None
        if sys.executable:
            python_exe = sys.executable.replace("\\", "/")

        cmd = _generate_command(args, skill_root, python_exe)

        print("")
        print("=" * 70)
        print("请复制以下命令到终端运行：")
        print("=" * 70)
        print("")
        print(cmd)
        print("")
        print("=" * 70)
        print("说明：")
        print(f"  - 发现 {len(pdf_paths)} 个 PDF 文件待处理")
        if args.dry_run:
            print("  - 干跑模式：只预览文件列表，不实际调用 OCR")
        if args.recursive:
            print("  - 递归模式：会扫描所有子目录")
        if args.no_resume:
            print("  - 不使用断点续跑")
        if args.limit:
            print(f"  - 限制处理数量：{args.limit}")
        if args.workers and args.workers > 1:
            print(f"  - 并发线程数：{args.workers}")
        print("=" * 70)
        return 0

    reports = _prepare_reports(pdf_paths, base_dir, out_root)
    state_manager = StateManager(state_file)

    max_retries = 3
    if not args.no_resume:
        state_manager.register_tasks([r.unique_id for r, _, _ in reports])
        reports = [
            (r, out_dir, uid)
            for r, out_dir, uid in reports
            if state_manager.is_pending_or_failed(uid, max_retries=max_retries)
        ]

    if args.dry_run:
        print("干跑模式（不执行 OCR）")
        for i, (report, out_dir, _) in enumerate(reports[:10], 1):
            print(f"{i:3}. {report.pdf_path} -> {out_dir}")
        if len(reports) > 10:
            print(f"... 还有 {len(reports) - 10} 个文件")
        return 0

    tokens_file = Path(__file__).parent.resolve() / "tokens.txt"
    tokens = _load_tokens_from_file(tokens_file)

    if not tokens:
        print("未配置有效的 tokens.txt，无法执行 OCR")
        return 1

    api_base = os.getenv("MINERU_API_BASE", "https://mineru.net/api/v4").strip()
    token_check_url = os.getenv(
        "MINERU_TOKEN_CHECK_URL",
        "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
    ).strip()
    available_tokens = []
    unavailable_tokens = []
    for t in tokens:
        ok, reason = _check_token_available(t, api_base, token_check_url)
        if ok:
            available_tokens.append(t)
        else:
            unavailable_tokens.append((t, reason))

    print(f"可用 Token: {len(available_tokens)} / {len(tokens)}")
    for t, reason in unavailable_tokens:
        print(f"不可用: {t[:8]}... | {reason}")

    if not available_tokens:
        print("没有可用 Token，停止执行 OCR")
        return 1

    token_manager = TokenQuotaManager(
        tokens=available_tokens,
        daily_page_limit=2000,
    )
    pdf_validator = PDFValidator()
    timeout_seconds = int(os.getenv("MINERU_TIMEOUT_SECONDS", "3600"))

    if args.workers <= 1:
        for report, out_dir, _ in reports:
            out_dir.mkdir(parents=True, exist_ok=True)
            result = _process_single(
                report,
                out_dir,
                state_manager,
                token_manager,
                pdf_validator,
                api_base,
                timeout_seconds,
            )
            if result.status == OCRStatus.SUCCESS:
                print(f"[成功] {report.pdf_path.name}")
            else:
                print(f"[失败] {report.pdf_path.name} | {result.error_message}")
        return 0

    success = 0
    failed = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for report, out_dir, _ in reports:
            out_dir.mkdir(parents=True, exist_ok=True)
            future = executor.submit(
                _process_single,
                report,
                out_dir,
                state_manager,
                token_manager,
                pdf_validator,
                api_base,
                timeout_seconds,
            )
            future_map[future] = report

        for future in as_completed(future_map):
            report = future_map[future]
            try:
                result = future.result()
                if result.status == OCRStatus.SUCCESS:
                    success += 1
                    print(f"[成功] {report.pdf_path.name}")
                else:
                    failed += 1
                    print(f"[失败] {report.pdf_path.name} | {result.error_message}")
            except Exception as e:
                failed += 1
                print(f"[异常] {report.pdf_path.name} | {e}")

    print(f"完成: 成功 {success} / 失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
