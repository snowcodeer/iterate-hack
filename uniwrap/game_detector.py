"""Detect game type and framework from codebase."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List


class GameDetector:
    """Detect game type and integration method."""
    
    def detect_game_type(self, repo_path: str) -> Dict:
        """Detect game type and framework.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            {
                'type': 'web'|'desktop'|'cli'|'api'|'unknown',
                'framework': 'react'|'vue'|'pygame'|'unity'|etc,
                'integration_method': 'browser'|'native'|'api'|'subprocess',
                'entry_point': path to main file,
                'config': framework-specific config
            }
        """
        repo_path = Path(repo_path)
        detected = {
            'type': 'unknown',
            'framework': 'unknown',
            'integration_method': 'browser',  # Default to browser for web games
            'entry_point': None,
            'config': {}
        }
        
        # Check for web games
        package_json = repo_path / 'package.json'
        if package_json.exists():
            detected['type'] = 'web'
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                deps = {**package_data.get('dependencies', {}), 
                       **package_data.get('devDependencies', {})}
                
                # Detect framework
                if 'react' in deps or any('react' in str(v) for v in deps.values()):
                    detected['framework'] = 'react'
                    detected['integration_method'] = 'browser'
                    # Find entry point
                    if (repo_path / 'src' / 'App.tsx').exists():
                        detected['entry_point'] = 'src/App.tsx'
                    elif (repo_path / 'src' / 'App.jsx').exists():
                        detected['entry_point'] = 'src/App.jsx'
                    elif (repo_path / 'App.tsx').exists():
                        detected['entry_point'] = 'App.tsx'
                    elif (repo_path / 'App.jsx').exists():
                        detected['entry_point'] = 'App.jsx'
                    
                elif 'vue' in deps:
                    detected['framework'] = 'vue'
                    detected['integration_method'] = 'browser'
                    
                elif 'angular' in deps or 'angular.json' in [f.name for f in repo_path.iterdir()]:
                    detected['framework'] = 'angular'
                    detected['integration_method'] = 'browser'
                    
                else:
                    # Vanilla JS web game
                    detected['framework'] = 'vanilla'
                    detected['integration_method'] = 'browser'
                    if (repo_path / 'index.html').exists():
                        detected['entry_point'] = 'index.html'
                
                # Check for dev server port
                scripts = package_data.get('scripts', {})
                if 'dev' in scripts:
                    # Try to extract port from dev script
                    dev_script = scripts['dev']
                    if '3000' in dev_script:
                        detected['config']['port'] = 3000
                    elif '5173' in dev_script:  # Vite default
                        detected['config']['port'] = 5173
                    else:
                        detected['config']['port'] = 3000  # Default
                else:
                    detected['config']['port'] = 3000
                    
            except (json.JSONDecodeError, KeyError):
                detected['framework'] = 'vanilla'
                detected['integration_method'] = 'browser'
        
        # Check for Python games (scan all .py files for pygame)
        py_files = list(repo_path.rglob('*.py'))[:20]  # Limit search
        pygame_file = None
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if 'import pygame' in content or 'from pygame' in content:
                    pygame_file = py_file
                    break
            except:
                continue

        if pygame_file:
            detected['type'] = 'desktop'
            detected['framework'] = 'pygame'
            detected['integration_method'] = 'native'
            detected['entry_point'] = str(pygame_file.relative_to(repo_path))
            # Detect game type for specialized controllers
            try:
                content = pygame_file.read_text(encoding='utf-8', errors='ignore').lower()
                if 'snake' in content or 'snake' in str(pygame_file).lower():
                    detected['config']['game_type'] = 'snake'
                elif 'pong' in content or 'pong' in str(pygame_file).lower():
                    detected['config']['game_type'] = 'pong'
                elif 'breakout' in content or 'brick' in content:
                    detected['config']['game_type'] = 'breakout'
                else:
                    detected['config']['game_type'] = 'generic'
            except:
                detected['config']['game_type'] = 'generic'
        elif (repo_path / 'main.py').exists() or (repo_path / '__main__.py').exists():
            detected['type'] = 'desktop'
            detected['integration_method'] = 'native'
            detected['framework'] = 'python'
            if (repo_path / 'main.py').exists():
                detected['entry_point'] = 'main.py'
            elif (repo_path / '__main__.py').exists():
                detected['entry_point'] = '__main__.py'
        
        # Check for Unity
        elif any((repo_path / f).exists() for f in ['.unity', 'Assets', 'ProjectSettings']):
            detected['type'] = 'desktop'
            detected['framework'] = 'unity'
            detected['integration_method'] = 'mlagents'  # Unity ML-Agents
        
        # Check for API/server games
        elif any((repo_path / f).exists() for f in ['server.py', 'app.py', 'main.go', 'server.js']):
            detected['type'] = 'api'
            detected['integration_method'] = 'api'
            
            if (repo_path / 'server.py').exists():
                detected['framework'] = 'flask'
            elif (repo_path / 'app.py').exists():
                detected['framework'] = 'fastapi'
            elif (repo_path / 'main.go').exists():
                detected['framework'] = 'go'
            elif (repo_path / 'server.js').exists():
                detected['framework'] = 'express'
        
        # Check if it's a URL (web game)
        if repo_path.as_uri().startswith('http'):
            detected['type'] = 'web'
            detected['framework'] = 'unknown'
            detected['integration_method'] = 'browser'
            detected['config']['url'] = str(repo_path)
        
        return detected

