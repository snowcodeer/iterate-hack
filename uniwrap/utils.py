"""Helper functions for file reading, JSON handling, and directory operations."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def read_file_safe(path: str) -> Optional[str]:
    """Safely read a file with encoding handling.
    
    Args:
        path: Path to the file to read
        
    Returns:
        File contents as string, or None if file cannot be read
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except (OSError, IOError, UnicodeDecodeError):
        return None


def get_directory_tree(path: str, max_depth: int = 3) -> str:
    """Generate a directory tree structure string.
    
    Args:
        path: Root directory path
        max_depth: Maximum depth to traverse
        
    Returns:
        Formatted tree structure as string
    """
    def _build_tree(dir_path: Path, prefix: str = "", depth: int = 0) -> List[str]:
        if depth > max_depth:
            return []
        
        lines = []
        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name))
            # Filter out common ignore patterns
            ignore_patterns = {'.git', '__pycache__', 'venv', 'node_modules', '.venv', 
                             'env', '.env', 'dist', 'build', '.pytest_cache', '.mypy_cache'}
            
            filtered_items = [item for item in items if item.name not in ignore_patterns]
            
            for i, item in enumerate(filtered_items):
                is_last = i == len(filtered_items) - 1
                current_prefix = "└── " if is_last else "├── "
                lines.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and depth < max_depth:
                    extension = "    " if is_last else "│   "
                    lines.extend(_build_tree(item, prefix + extension, depth + 1))
        except (PermissionError, OSError):
            pass
        
        return lines
    
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return ""
    
    lines = [str(root)]
    lines.extend(_build_tree(root))
    return "\n".join(lines)


def find_test_files(path: str) -> List[str]:
    """Discover test files in the codebase.
    
    Args:
        path: Root directory to search
        
    Returns:
        List of relative paths to test files
    """
    test_files = []
    root = Path(path)
    
    test_patterns = ['test_*.py', '*_test.py']
    
    for pattern in test_patterns:
        for test_file in root.rglob(pattern):
            # Skip venv and common ignore directories
            if any(ignore in test_file.parts for ignore in 
                   ['.git', '__pycache__', 'venv', 'node_modules', '.venv', 'env', '.env']):
                continue
            test_files.append(str(test_file.relative_to(root)))
    
    return sorted(test_files)


def find_entry_points(path: str) -> Dict:
    """Extract CLI/main entry points from the codebase.
    
    Args:
        path: Root directory to search
        
    Returns:
        Dictionary with entry point information
    """
    entry_points = {
        'main_files': [],
        'setup_py': None,
        'pyproject_toml': None,
        'package_json': None
    }
    
    root = Path(path)
    
    # Check for main.py or __main__.py
    for main_file in ['main.py', '__main__.py']:
        main_path = root / main_file
        if main_path.exists():
            entry_points['main_files'].append(main_file)
    
    # Check for setup.py
    setup_py = root / 'setup.py'
    if setup_py.exists():
        entry_points['setup_py'] = str(setup_py.relative_to(root))
    
    # Check for pyproject.toml
    pyproject = root / 'pyproject.toml'
    if pyproject.exists():
        entry_points['pyproject_toml'] = str(pyproject.relative_to(root))
    
    # Check for package.json (for Node.js projects)
    package_json = root / 'package.json'
    if package_json.exists():
        entry_points['package_json'] = str(package_json.relative_to(root))
    
    return entry_points


def clean_json_response(text: str) -> str:
    """Extract JSON from markdown code blocks or plain text.
    
    Args:
        text: Raw response text that may contain JSON in code blocks
        
    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks if present
    json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to find JSON object or array in the text
    json_obj_pattern = r'(\{.*\}|\[.*\])'
    match = re.search(json_obj_pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # If no JSON found, return original text (may be plain JSON)
    return text.strip()

