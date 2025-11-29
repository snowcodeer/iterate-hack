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

from uniwrap.env_designer import generate_env_specs
from uniwrap.repo_summarizer import summarize_repo
from uniwrap.spec_validator import validate_spec
from uniwrap.code_generator import generate_env_code


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
        help='Path to the target codebase repository'
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
        '--variants', '-n',
        type=int,
        default=3,
        help='Number of environment variants to generate in parallel (default: 3)'
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
    
    # Validate repo path
    repo_path = os.path.abspath(args.repo_path)
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
        # Step 1: Summarize repository
        print(f"Analyzing codebase at: {repo_path}")
        repo_summary = summarize_repo(repo_path)
        print("✓ Repository summarized")
        
        # Step 2: Generate environment specs in parallel
        print(f"Running {args.variants} parallel agents to generate environment specifications...")
        specs = generate_env_specs(
            repo_summary,
            num_variants=args.variants,
            model=args.model,
            api_key=api_key
        )
        print(f"✓ Generated {len(specs)} specification(s) from {args.variants} parallel agent(s)")
        
        # Step 3: Validate specs
        print("Validating specifications...")
        valid_specs = []
        invalid_count = 0
        for i, spec in enumerate(specs):
            is_valid, errors = validate_spec(spec)
            if is_valid:
                valid_specs.append(spec)
            else:
                invalid_count += 1
                print(f"  Warning: Spec {i+1} has validation errors:", file=sys.stderr)
                for error in errors:
                    print(f"    - {error}", file=sys.stderr)
        
        if invalid_count > 0:
            print(f"  ⚠ {invalid_count} specification(s) had validation errors", file=sys.stderr)
        
        if not valid_specs:
            print("Error: No valid specifications generated", file=sys.stderr)
            sys.exit(1)
        
        print(f"✓ {len(valid_specs)} valid specification(s)")
        
        # Step 4: Generate output files
        output_base_path = Path(args.output)
        
        # Extract repo name from path for subdirectory
        repo_name = Path(repo_path).name.replace('-', '_').replace('.', '_').lower()
        if repo_name.startswith('test_repo_'):
            repo_name = repo_name[10:]  # Remove test_repo_ prefix
        
        if args.format in ['code', 'both']:
            # Generate Python environment files in repo-specific subdirectory
            output_path = output_base_path / repo_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for the repo subdirectory
            init_file = output_path / '__init__.py'
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(f'"""Generated RL Environments for {repo_name}."""\n\n')
            
            env_names = []
            for i, spec in enumerate(valid_specs):
                env_name = spec.get('name', f'env_{i+1}').replace('-', '_')
                python_code = generate_env_code(spec, repo_path)
                
                python_file = output_path / f"{env_name}.py"
                with open(python_file, 'w', encoding='utf-8') as f:
                    f.write(python_code)
                
                # Add to __init__.py
                with open(init_file, 'a', encoding='utf-8') as f:
                    f.write(f"from .{env_name} import {env_name}\n")
                
                env_names.append(env_name)
                print(f"  ✓ Generated: {repo_name}/{python_file.name}")
            
            # Add __all__ to __init__.py
            with open(init_file, 'a', encoding='utf-8') as f:
                f.write(f'\n__all__ = {env_names}\n')
            
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
            
            print(f"✓ Python environment files written to: {output_path.absolute()}")
        
        if args.format in ['json', 'both']:
            # Generate JSON specs file
            json_output_path = output_base_path / repo_name if args.format == 'both' else Path(args.output)
            json_file = json_output_path / 'specs.json' if args.format == 'both' else json_output_path
            json_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(valid_specs, f, indent=2, ensure_ascii=False)
            
            print(f"✓ JSON specifications written to: {json_file.absolute()}")
        
        print(f"\nSummary: Generated {len(valid_specs)} environment(s) for {repo_name}")
        if args.format in ['code', 'both']:
            print(f"\nTo use the environments:")
            print(f"  from environments.{repo_name} import {valid_specs[0].get('name', 'env_1').replace('-', '_')}")
            print(f"  env = {valid_specs[0].get('name', 'env_1').replace('-', '_')}()")
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

