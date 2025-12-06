"""
GitHub persistence module for committing run data to the repository.

This allows run data to persist across Streamlit Cloud deployments by
automatically committing data files to GitHub after each run completes.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def commit_run_data_to_github(
    data_dir: str,
    run_id: str,
    github_token: Optional[str] = None,
    repo_owner: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> bool:
    """
    Commit run data files to GitHub using the GitHub API.

    Args:
        data_dir: Directory containing the data files
        run_id: The run ID to commit
        github_token: GitHub personal access token (or from GITHUB_TOKEN env var)
        repo_owner: Repository owner (or auto-detect from git remote)
        repo_name: Repository name (or auto-detect from git remote)

    Returns:
        True if successful, False otherwise
    """
    try:
        from github import Github
    except ImportError:
        logger.warning("PyGithub not installed. Install with: pip install PyGithub")
        return False

    github_token = github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning("⚠️ GITHUB_TOKEN not set. Cannot commit to GitHub.")
        logger.warning("To enable GitHub persistence:")
        logger.warning("  - Locally: export GITHUB_TOKEN=your_token")
        logger.warning("  - Streamlit Cloud: Add GITHUB_TOKEN in app secrets")
        return False

    # Try to auto-detect repo from git remote
    if not repo_owner or not repo_name:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Parse git@github.com:owner/repo.git or https://github.com/owner/repo.git
                if "github.com" in url:
                    parts = url.replace(".git", "").split("/")
                    if len(parts) >= 2:
                        repo_name = parts[-1]
                        repo_owner = parts[-2].split(":")[-1]
        except Exception as e:
            logger.debug(f"Could not auto-detect repo: {e}")

    if not repo_owner or not repo_name:
        logger.warning("⚠️ Could not determine repository owner/name.")
        logger.warning("Set GITHUB_REPO_OWNER and GITHUB_REPO_NAME environment variables.")
        logger.warning(f"Current values: repo_owner={repo_owner}, repo_name={repo_name}")
        return False

    try:
        logger.info(f"🔄 Connecting to GitHub repo: {repo_owner}/{repo_name}")
        g = Github(github_token)
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        logger.info(f"✅ Successfully connected to repo: {repo.full_name}")

        # Files to commit
        files_to_commit = []
        base_path = Path(data_dir)

        # Completions
        completions_file = base_path / "completions" / f"{run_id}.jsonl"
        if completions_file.exists():
            files_to_commit.append(("completions", completions_file))

        # Metrics
        metrics_file = base_path / "metrics" / f"{run_id}.jsonl"
        if metrics_file.exists():
            files_to_commit.append(("metrics", metrics_file))

        # Analysis
        analysis_file = base_path / "analysis" / f"{run_id}.jsonl"
        if analysis_file.exists():
            files_to_commit.append(("analysis", analysis_file))

        # Run progress log
        log_file = base_path / "logs" / f"run_{run_id}.json"
        if log_file.exists():
            files_to_commit.append(("logs", log_file))

        if not files_to_commit:
            logger.warning(f"⚠️ No files to commit for run_id={run_id}")
            logger.warning(f"Expected files in: {base_path}")
            return False

        logger.info(f"📦 Found {len(files_to_commit)} files to commit for run_id={run_id}")

        # Read file contents
        commit_files = {}
        for category, file_path in files_to_commit:
            git_path = f"data/{category}/{file_path.name}"
            logger.info(f"  📄 Reading: {git_path} ({file_path.stat().st_size} bytes)")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            commit_files[git_path] = content

        # Get current branch (usually 'main')
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            branch = result.stdout.strip() if result.returncode == 0 else "main"
        except Exception:
            branch = "main"

        # Get current tree SHA
        try:
            ref = repo.get_git_ref(f"heads/{branch}")
            base_sha = repo.get_git_commit(ref.object.sha).tree.sha
        except Exception as e:
            logger.warning(f"Could not get base SHA: {e}. Creating new commit.")
            base_sha = None

        # Create blobs
        logger.info(f"🔨 Creating Git blobs...")
        blobs = {}
        for path, content in commit_files.items():
            blob = repo.create_git_blob(content, "utf-8")
            blobs[path] = blob.sha
            logger.info(f"  ✅ Blob created: {path} (SHA: {blob.sha[:8]}...)")

        # Create tree
        from github import InputGitTreeElement

        tree_elements = []
        for path, blob_sha in blobs.items():
            tree_elements.append(
                InputGitTreeElement(
                    path=path,
                    mode="100644",
                    type="blob",
                    sha=blob_sha
                )
            )

        logger.info(f"🌳 Creating Git tree (branch: {branch})...")
        if base_sha:
            tree = repo.create_git_tree(tree_elements, base_tree=repo.get_git_tree(base_sha))
        else:
            tree = repo.create_git_tree(tree_elements)
        logger.info(f"  ✅ Tree created: {tree.sha[:8]}...")

        # Create commit
        logger.info(f"💾 Creating commit...")
        parent = repo.get_git_ref(f"heads/{branch}").object.sha
        commit_message = f"Add run data for {run_id}"
        commit = repo.create_git_commit(commit_message, tree, [parent])
        logger.info(f"  ✅ Commit created: {commit.sha[:8]}...")

        # Update branch reference
        logger.info(f"🚀 Updating branch reference...")
        ref.edit(commit.sha)

        logger.info(f"✅✅✅ Successfully committed run data for {run_id} to GitHub!")
        logger.info(f"     View commit: {repo.html_url}/commit/{commit.sha}")
        return True

    except Exception as e:
        logger.exception(f"Failed to commit run data to GitHub: {e}")
        return False

