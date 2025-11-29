"""Command-line interface for UniWrap."""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from uniwrap.env_designer import generate_env_spec, generate_pygame_env_code, generate_web_game_env_code
from uniwrap.repo_summarizer import summarize_repo
from uniwrap.spec_validator import validate_spec
from uniwrap.code_generator import generate_env_code
from uniwrap.game_detector import GameDetector
from uniwrap.evaluator import evaluate_environment


def main():
    """Main CLI entry point."""
    # Load .env file if python-dotenv is available
    if load_dotenv is not None:
        # Try to load .env from current directory or project root
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Also try current working directory
            load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="UniWrap: Generate RL environment specifications from any codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'repo_path',
        type=str,
        help='Path to the target codebase repository, GitHub repo URL, or web game URL'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='environments',
        help='Output directory for Python environment files (default: environments/)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['code', 'json', 'both'],
        default='code',
        help='Output format: code (Python files), json (specs), or both (default: code)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation agent to assess wrapper quality after generation'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-5-20250929',
        help='Claude model to use (default: claude-sonnet-4-5-20250929)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Anthropic API key (default: uses ANTHROPIC_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Handle URLs (GitHub repos or web game URLs)
    repo_path = args.repo_path
    is_url = repo_path.startswith('http://') or repo_path.startswith('https://')
    is_github = 'github.com' in repo_path.lower() if is_url else False
    
    if is_github:
        # Clone GitHub repo
        import tempfile
        import subprocess
        print(f"Cloning GitHub repository: {repo_path}")
        # Extract repo name from URL for output naming
        github_repo_name = repo_path.rstrip('/').split('/')[-1].replace('.git', '')
        temp_dir = tempfile.mkdtemp(prefix='uniwrap_')
        try:
            subprocess.run(['git', 'clone', repo_path, temp_dir], check=True, capture_output=True)
            repo_path = temp_dir
            print(f"✓ Repository cloned to: {temp_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to clone repository: {e}", file=sys.stderr)
            sys.exit(1)
    elif is_url:
        # Web game URL - this is a playable game, not source code
        print(f"Detected web game URL: {repo_path}")
        web_game_url = repo_path  # Save for later
    else:
        # Local path
        repo_path = os.path.abspath(repo_path)
        if not os.path.exists(repo_path):
            print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
            sys.exit(1)
        
        if not os.path.isdir(repo_path):
            print(f"Error: Repository path is not a directory: {repo_path}", file=sys.stderr)
            sys.exit(1)
    
    # Validate API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY not found. Set it as an environment variable "
            "or pass it with --api-key",
            file=sys.stderr
        )
        sys.exit(1)
    
    try:
        # Handle web game URLs specially (not GitHub, just a playable game URL)
        if is_url and not is_github:
            print("Fetching web game page...")

            # Fetch page info using requests or similar
            try:
                import urllib.request
                with urllib.request.urlopen(repo_path, timeout=30) as response:
                    page_html = response.read().decode('utf-8', errors='ignore')[:50000]  # Limit size
            except Exception as e:
                print(f"Warning: Could not fetch page: {e}")
                page_html = f"URL: {repo_path}\nCould not fetch page content."

            # Create page info summary for Claude
            page_info = f"""
URL: {repo_path}

This appears to be a web-based game. Common patterns for browser games:
- Chrome Dino game: canvas-based, space to jump, down to duck
- Flappy Bird clones: space/click to flap
- Simple arcade games: arrow keys or WASD for movement

Page HTML preview (first 5000 chars):
{page_html[:5000]}
"""
            print("✓ Page info gathered")

            # Extract game name from URL
            from urllib.parse import urlparse
            parsed = urlparse(repo_path)
            repo_name = parsed.netloc.replace('.', '_').replace('-', '_').lower()
            if repo_name.startswith('www_'):
                repo_name = repo_name[4:]

            print("Generating web game environment (Claude will create Playwright wrapper)...")

            # Generate web game environment
            python_code = generate_web_game_env_code(
                game_url=repo_path,
                page_info=page_info,
                model=args.model,
                api_key=api_key
            )
            print("✓ Environment code generated")

            # Extract class name from generated code
            import re
            class_match = re.search(r'class\s+(\w+)\s*\(', python_code)
            env_name = class_match.group(1) if class_match else f"{repo_name}Env"

            output_base_path = Path(args.output)

            if args.format in ['code', 'both']:
                # Write the generated environment
                output_path = output_base_path / repo_name
                output_path.mkdir(parents=True, exist_ok=True)

                # Create __init__.py for the repo subdirectory
                init_file = output_path / '__init__.py'
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Generated RL Environment for {repo_name}."""\n\n')

                python_file = output_path / f"{env_name.lower()}.py"
                with open(python_file, 'w', encoding='utf-8') as f:
                    f.write(python_code)

                # Add to __init__.py
                with open(init_file, 'a', encoding='utf-8') as f:
                    f.write(f"from .{env_name.lower()} import {env_name}\n")
                    f.write(f'\n__all__ = ["{env_name}"]\n')

                print(f"  ✓ Generated: {repo_name}/{python_file.name}")

                # Create top-level __init__.py for environments package
                top_init = output_base_path / '__init__.py'
                if not top_init.exists():
                    with open(top_init, 'w', encoding='utf-8') as f:
                        f.write('"""UniWrap Generated Environments."""\n\n')

                # Add the repo module to top-level __init__.py
                with open(top_init, 'r', encoding='utf-8') as f:
                    content = f.read()

                import_line = f"from . import {repo_name}\n"
                if import_line not in content:
                    with open(top_init, 'a', encoding='utf-8') as f:
                        if not content.endswith('\n'):
                            f.write('\n')
                        f.write(import_line)

                print(f"✓ Python environment file written to: {output_path.absolute()}")

            print(f"\nSummary: Generated 1 environment for {repo_name}")
            print(f"\nTo use the environment:")
            print(f"  from environments.{repo_name} import {env_name}")
            print(f"  env = {env_name}()")
            print(f"  obs, info = env.reset()")
            print(f"  action = env.action_space.sample()")
            print(f"  obs, reward, terminated, truncated, info = env.step(action)")

            print(f"\nNote: This environment requires Playwright. Install with:")
            print(f"  pip install playwright && playwright install chromium")

            sys.exit(0)

        # Step 1: Detect game type (for GitHub repos or local paths)
        print("Detecting game type...")
        detector = GameDetector()
        game_info = detector.detect_game_type(repo_path)
        print(f"✓ Detected: {game_info['type']} game ({game_info['framework']})")
        print(f"  Integration method: {game_info['integration_method']}")

        # Step 2: Summarize repository
        print(f"Analyzing codebase...")
        repo_summary = summarize_repo(repo_path)
        print("✓ Repository summarized")

        # Extract repo name from path for subdirectory
        if is_github:
            # Use the GitHub repo name, not temp dir name
            repo_name = github_repo_name.replace('-', '_').replace('.', '_').lower()
        else:
            repo_name = Path(repo_path).name.replace('-', '_').replace('.', '_').lower()
            if repo_name.startswith('test_repo_'):
                repo_name = repo_name[10:]  # Remove test_repo_ prefix
            if repo_name.startswith('uniwrap_'):
                repo_name = repo_name[8:]  # Remove uniwrap_ prefix from temp dirs

        output_base_path = Path(args.output)

        # Handle pygame games specially - generate complete environment code directly
        if game_info.get('framework') == 'pygame' and game_info.get('entry_point'):
            print("Generating pygame environment (Claude will analyze game code)...")

            # Read the game source code
            game_file_path = Path(repo_path) / game_info['entry_point']
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_code = f.read()

            # Generate complete environment code using Claude
            python_code = generate_pygame_env_code(
                game_code=game_code,
                repo_summary=repo_summary,
                model=args.model,
                api_key=api_key
            )
            print("✓ Environment code generated")

            # Extract class name from generated code
            import re
            class_match = re.search(r'class\s+(\w+)\s*\(', python_code)
            env_name = class_match.group(1) if class_match else f"{repo_name}_env"

            if args.format in ['code', 'both']:
                # Write the generated environment
                output_path = output_base_path / repo_name
                output_path.mkdir(parents=True, exist_ok=True)

                # Create __init__.py for the repo subdirectory
                init_file = output_path / '__init__.py'
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Generated RL Environment for {repo_name}."""\n\n')

                python_file = output_path / f"{env_name.lower()}.py"
                with open(python_file, 'w', encoding='utf-8') as f:
                    f.write(python_code)

                # Add to __init__.py
                with open(init_file, 'a', encoding='utf-8') as f:
                    f.write(f"from .{env_name.lower()} import {env_name}\n")
                    f.write(f'\n__all__ = ["{env_name}"]\n')

                print(f"  ✓ Generated: {repo_name}/{python_file.name}")

                # Create top-level __init__.py for environments package
                top_init = output_base_path / '__init__.py'
                if not top_init.exists():
                    with open(top_init, 'w', encoding='utf-8') as f:
                        f.write('"""UniWrap Generated Environments."""\n\n')

                # Add the repo module to top-level __init__.py
                with open(top_init, 'r', encoding='utf-8') as f:
                    content = f.read()

                import_line = f"from . import {repo_name}\n"
                if import_line not in content:
                    with open(top_init, 'a', encoding='utf-8') as f:
                        if not content.endswith('\n'):
                            f.write('\n')
                        f.write(import_line)

                print(f"✓ Python environment file written to: {output_path.absolute()}")

                # Step 6: Evaluate environment if requested
                if args.evaluate:
                    print("\nRunning evaluation agent...")
                    eval_results = evaluate_environment(
                        env_path=python_file,
                        env_name=env_name,
                        game_info=game_info
                    )
                    print(f"✓ Evaluation complete")
                    print(f"  Quality Score: {eval_results.get('quality_score', 'N/A')}")
                    print(f"  Issues Found: {len(eval_results.get('issues', []))}")
                    if eval_results.get('issues'):
                        for issue in eval_results['issues']:
                            print(f"    - {issue}")

            print(f"\nSummary: Generated 1 environment for {repo_name}")
            if args.format in ['code', 'both']:
                print(f"\nTo use the environment:")
                print(f"  from environments.{repo_name} import {env_name}")
                print(f"  env = {env_name}()")
                print(f"  obs, info = env.reset()")
                print(f"  action = env.action_space.sample()")
                print(f"  obs, reward, terminated, truncated, info = env.step(action)")

        else:
            # Standard flow for non-pygame games
            # Step 3: Generate single environment spec
            print("Generating environment specification...")
            spec = generate_env_spec(
                repo_summary,
                game_info=game_info,
                model=args.model,
                api_key=api_key
            )
            print("✓ Environment specification generated")

            # Step 4: Validate spec
            print("Validating specification...")
            is_valid, errors = validate_spec(spec)
            if not is_valid:
                print("  Warning: Specification has validation errors:", file=sys.stderr)
                for error in errors:
                    print(f"    - {error}", file=sys.stderr)
                print("  Continuing anyway...", file=sys.stderr)

            print("✓ Specification validated")

            if args.format in ['code', 'both']:
                # Generate Python environment file in repo-specific subdirectory
                output_path = output_base_path / repo_name
                output_path.mkdir(parents=True, exist_ok=True)

                # Create __init__.py for the repo subdirectory
                init_file = output_path / '__init__.py'
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Generated RL Environment for {repo_name}."""\n\n')

                env_name = spec.get('name', 'generated_env').replace('-', '_')
                python_code = generate_env_code(spec, repo_path, game_info=game_info)

                python_file = output_path / f"{env_name}.py"
                with open(python_file, 'w', encoding='utf-8') as f:
                    f.write(python_code)

                # Add to __init__.py
                with open(init_file, 'a', encoding='utf-8') as f:
                    f.write(f"from .{env_name} import {env_name}\n")
                    f.write(f'\n__all__ = ["{env_name}"]\n')

                print(f"  ✓ Generated: {repo_name}/{python_file.name}")

                # Create top-level __init__.py for environments package
                top_init = output_base_path / '__init__.py'
                if not top_init.exists():
                    with open(top_init, 'w', encoding='utf-8') as f:
                        f.write('"""UniWrap Generated Environments."""\n\n')

                # Add the repo module to top-level __init__.py
                with open(top_init, 'r', encoding='utf-8') as f:
                    content = f.read()

                import_line = f"from . import {repo_name}\n"
                if import_line not in content:
                    with open(top_init, 'a', encoding='utf-8') as f:
                        if not content.endswith('\n'):
                            f.write('\n')
                        f.write(import_line)

                print(f"✓ Python environment file written to: {output_path.absolute()}")

                # Step 6: Evaluate environment if requested
                if args.evaluate:
                    print("\nRunning evaluation agent...")
                    eval_results = evaluate_environment(
                        env_path=python_file,
                        env_name=env_name,
                        game_info=game_info
                    )
                    print(f"✓ Evaluation complete")
                    print(f"  Quality Score: {eval_results.get('quality_score', 'N/A')}")
                    print(f"  Issues Found: {len(eval_results.get('issues', []))}")
                    if eval_results.get('issues'):
                        for issue in eval_results['issues']:
                            print(f"    - {issue}")

            if args.format in ['json', 'both']:
                # Generate JSON spec file
                json_output_path = output_base_path / repo_name if args.format == 'both' else Path(args.output)
                json_file = json_output_path / 'spec.json' if args.format == 'both' else json_output_path
                json_file.parent.mkdir(parents=True, exist_ok=True)

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(spec, f, indent=2, ensure_ascii=False)

                print(f"✓ JSON specification written to: {json_file.absolute()}")

            print(f"\nSummary: Generated 1 environment for {repo_name}")
            if args.format in ['code', 'both']:
                print(f"\nTo use the environment:")
                print(f"  from environments.{repo_name} import {env_name}")
                print(f"  env = {env_name}()")
                print(f"  obs, info = env.reset()")
                print(f"  action = env.action_space.sample()")
                print(f"  obs, reward, terminated, truncated, info = env.step(action)")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

