#!/usr/bin/env python3
"""CLI for testing generated environments with agents."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uniwrap.test_agent import test_all_environments


def main():
    """Main entry point for testing environments."""
    parser = argparse.ArgumentParser(
        description="Test generated RL environments with agents and generate evaluation reports"
    )
    
    parser.add_argument(
        'repo_name',
        type=str,
        help='Repository name (subdirectory in environments/)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=10,
        help='Number of episodes to run per environment (default: 10)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualizations after testing'
    )
    
    args = parser.parse_args()
    
    # Test all environments for the repo
    test_all_environments(args.repo_name, args.episodes, args.visualize)


if __name__ == '__main__':
    main()

