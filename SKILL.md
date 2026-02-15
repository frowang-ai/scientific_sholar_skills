---
name: ocr-mineru
description: 自包含的 MinerU OCR Skill，对单个 PDF/目录/递归目录执行 OCR，支持多 Token 轮换与断点续跑。
---

# MinerU OCR Skill

## 功能说明
- 单个 PDF OCR
- 目录批量 OCR（可选递归）
- 多 Token 轮换（按日页数限额）
- 断点续跑（状态持久化）
- 干跑预览（不实际调用 API）

## 目录结构
- 入口脚本：`scripts/ocr_mineru_cli.py`
- 状态文件：`state/ocr_skill_state.jsonl`

## 使用方式
1) 单文件
```powershell
.\.venv\Scripts\python.exe .\scripts\ocr_mineru_cli.py --pdf "path\to\file.pdf" --out "data\ocr_out"
```

2) 目录批量
```powershell
.\.venv\Scripts\python.exe .\scripts\ocr_mineru_cli.py --dir "raw_data\pdf_2023" --out "data\ocr_out"
```

3) 递归目录
```powershell
.\.venv\Scripts\python.exe .\scripts\ocr_mineru_cli.py --dir "raw_data" --recursive --out "data\ocr_out"
```

4) 干跑预览
```powershell
.\.venv\Scripts\python.exe .\scripts\ocr_mineru_cli.py --dir "raw_data" --recursive --dry-run
```

## 参数说明
- `--pdf`：单个 PDF 文件（与 `--dir` 互斥）
- `--dir`：包含 PDF 的目录（与 `--pdf` 互斥）
- `--recursive`：递归扫描子目录
- `--out`：输出根目录（默认输入目录）
- `--dry-run`：只预览，不调用 OCR
- `--limit`：最多处理文件数量
- `--workers`：并发线程数（默认 1）
- `--no-resume`：不使用断点续跑

## 输出结构约定
输出目录按输入相对路径映射：
- 单文件：`out_root/<pdf_stem>/`
- 目录：`out_root/<相对路径>/<pdf_stem>/`

## Token 配置
从 `scripts/tokens.txt` 读取（每行一个 Token；支持 `#` 注释行）。
脚本启动时会先检测每个 Token 是否可用，仅使用可用 Token，并输出可用数量。

API 配置可用环境变量：
- `MINERU_API_BASE`：API Base（默认 `https://mineru.net/api/v4`）
- `MINERU_TIMEOUT_SECONDS`：超时秒数（默认 3600）
- `MINERU_TOKEN_CHECK_URL`：Token 检测用示例 PDF URL（默认 `https://cdn-mineru.openxlab.org.cn/demo/example.pdf`）

## 测试
```powershell
```