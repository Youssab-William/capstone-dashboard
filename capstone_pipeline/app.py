import argparse
import json
import re
import logging
import os
from pathlib import Path
from typing import List
from datetime import datetime, timezone
from .storage.repository import JsonlStorage
from .metrics.metrics_engine import ScriptMetricsEngine
from .analysis.analysis_engine import ScriptAnalysisEngine
from .workflow.pipeline import Pipeline
from .domain import TaskSpec
from .providers.deepseek import DeepSeekProvider
from .providers.gpt import GPTProvider
from .providers.claude import ClaudeProvider
from .providers.gemini import GeminiProvider
from .runner.progress import RunProgressTracker


def parse_tasks_file(path: str) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for row in data:
        task_id = row.get('TaskID', '')
        tone = row.get('PromptTone', '')
        prompt = row.get('PromptText', '')
        m = re.match(r"^(.*)Task\d+$", task_id)
        prefix = (m.group(1) if m else '')
        category = re.sub(r"[^A-Za-z]", "", prefix)
        tasks.append(TaskSpec(id=task_id, category=category, tone=tone, prompt=prompt))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(prog="capstone-pipeline")
    parser.add_argument("command", choices=["run", "collect", "metrics", "analyze"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--prompts_file", default="prompts.txt")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--log_file", default="")
    parser.add_argument("--keys_file", default="")
    args = parser.parse_args()
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(fh)
    logger = logging.getLogger("capstone_pipeline")
    logger.info(f"command={args.command} data_dir={args.data_dir} run_id={args.run_id}")
    if args.keys_file:
        try:
            with open(args.keys_file, 'r', encoding='utf-8') as f:
                keys = json.load(f)
            def set_key(env_name: str, value: str):
                if value:
                    os.environ[env_name] = value
                    logger.info(f"loaded_key {env_name}=")
            set_key('DEEPSEEK_API_KEY', keys.get('deepseek_api_key') or keys.get('DEEPSEEK_API_KEY'))
            set_key('OPENAI_API_KEY', keys.get('openai_api_key') or keys.get('OPENAI_API_KEY'))
            set_key('ANTHROPIC_API_KEY', keys.get('anthropic_api_key') or keys.get('ANTHROPIC_API_KEY'))
            set_key('GEMINI_API_KEY', keys.get('gemini_api_key') or keys.get('GEMINI_API_KEY'))
            set_key('GOOGLE_API_KEY', keys.get('google_api_key') or keys.get('GOOGLE_API_KEY'))
            logger.info("keys_file_loaded")
        except Exception as e:
            logger.warning(f"keys_file_error {e}")
    storage = JsonlStorage(args.data_dir)
    metrics = ScriptMetricsEngine()
    analysis = ScriptAnalysisEngine()
    pipeline = Pipeline(storage, metrics, analysis)
    tasks: List[TaskSpec] = parse_tasks_file(args.prompts_file)
    providers = [DeepSeekProvider(), GPTProvider(), ClaudeProvider(), GeminiProvider()]
    if args.command == "run":
        logger.info("starting full pipeline")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        progress = RunProgressTracker(args.data_dir, run_id)
        res = pipeline.run_all(tasks, providers, run_id=run_id, progress=progress)
        logger.info("pipeline completed run_id=%s", res.get("run_id", ""))
    elif args.command == "collect":
        logger.info("starting collect phase")
        pipeline.run_all(tasks, providers)
        logger.info("collect phase completed")
    elif args.command == "metrics":
        if not args.run_id:
            logger.error("metrics command requires --run_id")
            return
        completions = storage.load_rows("completions", args.run_id)
        from .domain import CompletionRecord
        def to_cr(r):
            return CompletionRecord(
                run_id=r.get("run_id",""),
                task_id=r.get("task_id",""),
                category=r.get("category",""),
                model=r.get("model",""),
                version=r.get("version",""),
                tone=r.get("tone",""),
                prompt=r.get("prompt",""),
                response_text=r.get("response_text",""),
                usage=r.get("usage"),
                created_at=datetime.fromisoformat(r.get("created_at")) if r.get("created_at") else datetime.now(timezone.utc),
            )
        crs = [to_cr(r) for r in completions]
        logger.info(f"computing metrics rows count={len(crs)} run_id={args.run_id}")
        rows = metrics.compute(crs)
        storage.save_rows(rows, "metrics", args.run_id)
        logger.info("metrics phase completed")
    elif args.command == "analyze":
        if not args.run_id:
            logger.error("analyze command requires --run_id")
            return
        metrics_rows = storage.load_rows("metrics", args.run_id)
        logger.info(f"analyzing metrics_rows count={len(metrics_rows)} run_id={args.run_id}")
        artifacts = analysis.analyze(metrics_rows)
        storage.save_rows([artifacts], "analysis", args.run_id)
        logger.info("analysis phase completed")


if __name__ == "__main__":
    main()
