"""Codebase analysis and summarization for LLM consumption."""

import os
from pathlib import Path
from typing import List

from uniwrap.utils import (
    read_file_safe,
    get_directory_tree,
    find_test_files,
    find_entry_points
)


def summarize_repo(repo_path: str, max_summary_length: int = 3000) -> str:
    """Summarize a codebase into LLM-friendly text.
    
    Args:
        repo_path: Path to the repository root
        max_summary_length: Maximum length of summary in characters
        
    Returns:
        Formatted summary string
    """
    repo_path = os.path.abspath(repo_path)
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    summary_parts = []
    
    # 1. README.md content
    readme_path = os.path.join(repo_path, "README.md")
    readme_content = read_file_safe(readme_path)
    if readme_content:
        summary_parts.append("=== README.md ===")
        # Truncate README if too long
        if len(readme_content) > 1000:
            readme_content = readme_content[:1000] + "\n... (truncated)"
        summary_parts.append(readme_content)
        summary_parts.append("")
    
    # 2. Directory tree structure
    summary_parts.append("=== Directory Structure ===")
    tree = get_directory_tree(repo_path, max_depth=3)
    if tree:
        summary_parts.append(tree)
    summary_parts.append("")
    
    # 3. Entry points
    entry_points = find_entry_points(repo_path)
    summary_parts.append("=== Entry Points ===")
    if entry_points['main_files']:
        summary_parts.append(f"Main files: {', '.join(entry_points['main_files'])}")
    if entry_points['setup_py']:
        summary_parts.append(f"Setup.py: {entry_points['setup_py']}")
    if entry_points['pyproject_toml']:
        summary_parts.append(f"Pyproject.toml: {entry_points['pyproject_toml']}")
    if entry_points['package_json']:
        summary_parts.append(f"Package.json: {entry_points['package_json']}")
    summary_parts.append("")
    
    # 4. Test files
    test_files = find_test_files(repo_path)
    if test_files:
        summary_parts.append("=== Test Files ===")
        # Limit to first 10 test files
        for test_file in test_files[:10]:
            summary_parts.append(f"  - {test_file}")
        if len(test_files) > 10:
            summary_parts.append(f"  ... and {len(test_files) - 10} more test files")
        summary_parts.append("")
    
    # 5. Key source files (main Python modules, excluding tests)
    summary_parts.append("=== Key Source Files ===")
    key_files = _find_key_source_files(repo_path)
    for key_file in key_files[:15]:  # Limit to 15 files
        content = read_file_safe(os.path.join(repo_path, key_file))
        if content:
            # Include first 200 chars of each file
            preview = content[:200].replace('\n', ' ')
            summary_parts.append(f"{key_file}: {preview}...")
    summary_parts.append("")
    
    full_summary = "\n".join(summary_parts)
    
    # Truncate if too long
    if len(full_summary) > max_summary_length:
        full_summary = full_summary[:max_summary_length] + "\n... (summary truncated)"
    
    return full_summary


def _find_key_source_files(repo_path: str) -> List[str]:
    """Find key source files in the repository.
    
    Args:
        repo_path: Root directory path
        
    Returns:
        List of relative file paths
    """
    key_files = []
    root = Path(repo_path)
    
    # Common patterns for important files
    important_patterns = [
        '*.py',
        '*.js',
        '*.ts',
        '*.java',
        '*.go',
        '*.rs',
        '*.cpp',
        '*.c',
    ]
    
    ignore_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.venv', 
                   'env', '.env', 'dist', 'build', '.pytest_cache', 
                   '.mypy_cache', 'tests', 'test'}
    
    for pattern in important_patterns:
        for file_path in root.rglob(pattern):
            # Skip ignored directories
            if any(ignore in file_path.parts for ignore in ignore_dirs):
                continue
            
            # Skip test files (already handled separately)
            if 'test' in file_path.name.lower() and file_path.suffix == '.py':
                continue
            
            rel_path = str(file_path.relative_to(root))
            key_files.append(rel_path)
    
    # Prioritize files in root or common source directories
    def priority_key(path: str):
        parts = Path(path).parts
        depth = len(parts)
        # Prefer files at root or in src/ directories
        if depth == 1:
            return (0, path)
        elif 'src' in parts:
            return (1, path)
        else:
            return (2, path)
    
    key_files.sort(key=priority_key)
    return key_files[:20]  # Return top 20 files

