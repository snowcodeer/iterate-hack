#!/usr/bin/env python3
"""CLI for generating visualizations from evaluation results."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from uniwrap.visualize import generate_visualizations


def main():
    """Main entry point for visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations and comparison plots from evaluation results"
    )
    
    parser.add_argument(
        'repo_name',
        type=str,
        help='Repository name (subdirectory in environments/)'
    )
    
    args = parser.parse_args()
    
    # Generate visualizations
    generate_visualizations(args.repo_name)


if __name__ == '__main__':
    main()

