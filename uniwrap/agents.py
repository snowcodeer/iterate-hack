"""
Two-Agent Architecture for UniWrap

Agent 1: Environment Generator
- Browses game URL to understand mechanics
- Generates initial environment code
- Improves code based on feedback from Agent 2

Agent 2: Trainer/Supervisor
- Analyzes environment code for issues
- Decides whether to train or request improvements
- Trains PPO agent and evaluates performance
- Communicates with Agent 1 for improvements
- Loops until quality threshold met
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from uniwrap.llm_client import ClaudeClient


# =============================================================================
# Tool Definitions
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None


def tool_browse_game(url: str) -> ToolResult:
    """Browse a game URL and extract information about how it works."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ToolResult(False, None, "requests/beautifulsoup not installed")

    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else ""

        # Get text content
        text_content = soup.get_text(separator=' ', strip=True)[:3000]

        # Find game instructions
        instructions = []
        keywords = ['press', 'click', 'tap', 'space', 'arrow', 'control',
                   'play', 'start', 'jump', 'dodge', 'avoid', 'move', 'wasd']
        for tag in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'li']):
            text = tag.get_text(strip=True).lower()
            if any(word in text for word in keywords):
                instructions.append(tag.get_text(strip=True)[:200])

        # Check for canvas
        has_canvas = soup.find('canvas') is not None
        canvas_ids = [c.get('id', '') for c in soup.find_all('canvas')]

        # Check for common game frameworks in scripts
        scripts = soup.find_all('script')
        inline_scripts_preview = ""
        for script in scripts[:5]:
            if script.string and len(script.string) > 50:
                inline_scripts_preview += script.string[:500] + "\n...\n"

        return ToolResult(True, {
            "url": url,
            "title": title,
            "has_canvas": has_canvas,
            "canvas_ids": canvas_ids[:3],
            "instructions": instructions[:10],
            "text_preview": text_content[:2000],
            "inline_scripts_preview": inline_scripts_preview[:1500]
        })
    except Exception as e:
        return ToolResult(False, None, str(e))


def tool_generate_environment(game_info: Dict, game_type: str, hints: str = "") -> ToolResult:
    """Generate a Gymnasium environment from game info."""
    from uniwrap.env_designer import generate_environment_code

    try:
        code = generate_environment_code(
            game_info=game_info,
            game_type=game_type,
            hints=hints
        )
        return ToolResult(True, {"code": code})
    except Exception as e:
        return ToolResult(False, None, str(e))


def tool_save_environment(env_name: str, code: str) -> ToolResult:
    """Save environment code to disk."""
    try:
        env_dir = Path(f"environments/{env_name}")
        env_dir.mkdir(parents=True, exist_ok=True)

        # Determine class name
        import re
        class_match = re.search(r'class\s+(\w+Env)\s*\(', code)
        class_name = class_match.group(1) if class_match else "GameEnv"

        # Write environment file
        env_file = env_dir / f"{class_name.lower().replace('env', '')}env.py"
        env_file.write_text(code)

        # Write __init__.py
        init_file = env_dir / "__init__.py"
        init_file.write_text(f'from .{env_file.stem} import {class_name}\n\n__all__ = ["{class_name}"]\n')

        return ToolResult(True, {
            "env_name": env_name,
            "env_path": str(env_file),
            "class_name": class_name
        })
    except Exception as e:
        return ToolResult(False, None, str(e))


def tool_analyze_code(code: str) -> ToolResult:
    """Analyze environment code for potential issues without training."""
    issues = []

    # Check for common problems
    if "def step(" not in code:
        issues.append({"severity": "critical", "issue": "Missing step() method"})
    if "def reset(" not in code:
        issues.append({"severity": "critical", "issue": "Missing reset() method"})
    if "observation_space" not in code:
        issues.append({"severity": "critical", "issue": "Missing observation_space"})
    if "action_space" not in code:
        issues.append({"severity": "critical", "issue": "Missing action_space"})

    # Check reward function
    if "reward" not in code.lower():
        issues.append({"severity": "critical", "issue": "No reward signal found"})
    elif code.count("reward") < 3:
        issues.append({"severity": "major", "issue": "Reward logic seems too simple"})

    # Check for game over detection
    if "_is_game_over" not in code and "terminated" not in code:
        issues.append({"severity": "major", "issue": "No clear game-over detection"})

    # Check for grace period in web games
    if "playwright" in code.lower() or "browser" in code.lower():
        if "grace" not in code.lower() and "steps_taken < 30" not in code:
            issues.append({"severity": "major", "issue": "Web game missing grace period for game-over detection"})
        if "identical_frame" in code and "< 30" not in code and ">= 30" not in code:
            issues.append({"severity": "major", "issue": "Frame freeze threshold may be too aggressive"})

    # Check for pygame font issues
    if "pygame" in code.lower():
        if "pygame.font" in code and "try:" not in code:
            issues.append({"severity": "minor", "issue": "Pygame font not wrapped in try/except (Python 3.14 issue)"})

    # Syntax check
    try:
        import ast
        ast.parse(code)
    except SyntaxError as e:
        issues.append({"severity": "critical", "issue": f"Syntax error: {e}"})

    return ToolResult(True, {
        "issues": issues,
        "has_critical": any(i["severity"] == "critical" for i in issues),
        "issue_count": len(issues)
    })


def tool_train_agent(env_name: str, timesteps: int = 5000) -> ToolResult:
    """Train a PPO agent on the environment."""
    try:
        import sys
        import importlib

        # Clear module cache
        for key in list(sys.modules.keys()):
            if key.startswith(f'environments.{env_name}'):
                del sys.modules[key]

        # Import environment
        env_module = importlib.import_module(f'environments.{env_name}')
        env_class_name = None
        for name in dir(env_module):
            if name.endswith('Env') and not name.startswith('_'):
                env_class_name = name
                break

        if not env_class_name:
            return ToolResult(False, None, "Could not find environment class")

        env_class = getattr(env_module, env_class_name)

        # Train
        from uniwrap.rl_agent import RLAgent
        agent = RLAgent(env_class, headless=True)

        episode_rewards = []
        episode_lengths = []

        def progress_callback(data):
            if data['type'] == 'episode_complete':
                episode_rewards.append(data['reward'])
                episode_lengths.append(data['length'])

        agent.train(total_timesteps=timesteps, progress_callback=progress_callback)
        agent.close()

        import numpy as np
        return ToolResult(True, {
            "episodes": len(episode_rewards),
            "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0,
            "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0,
            "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0,
            "reward_std": float(np.std(episode_rewards)) if episode_rewards else 0,
            "avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        })
    except Exception as e:
        import traceback
        return ToolResult(False, None, f"{str(e)}\n{traceback.format_exc()}")


def tool_improve_environment(code: str, issues: List[Dict], game_info: Dict) -> ToolResult:
    """Request improved environment code based on issues."""
    from uniwrap.env_supervisor import improve_environment

    try:
        # Format analysis for improvement
        analysis = {
            "issues": issues,
            "quality_score": max(1, 10 - len(issues) * 2),
            "game_info": json.dumps(game_info, indent=2) if game_info else "N/A"
        }

        result = improve_environment(
            env_code=code,
            analysis=analysis,
            user_feedback="Fix all identified issues. Ensure the environment is trainable."
        )

        return ToolResult(True, {
            "improved_code": result['code'],
            "changelog": result['changelog']
        })
    except Exception as e:
        return ToolResult(False, None, str(e))


# =============================================================================
# Agent Definitions
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are Agent 1: The Environment Generator.

Your role is to create and improve Gymnasium RL environments from games.

You have these tools:
- browse_game: Fetch and analyze a game URL to understand mechanics
- generate_environment: Create initial environment code
- save_environment: Save environment to disk
- improve_environment: Improve code based on feedback

When asked to create an environment:
1. First browse the game URL to understand how it works
2. Identify the game type (web game vs pygame)
3. Generate environment code
4. Save it to disk

When asked to improve an environment:
1. Review the issues provided
2. Generate improved code that fixes the issues
3. Save the updated version

Always explain what you're doing and why."""


SUPERVISOR_SYSTEM_PROMPT = """You are Agent 2: The Trainer/Supervisor.

Your role is to ensure environments are trainable and improve them iteratively.

You have these tools:
- analyze_code: Check environment code for issues (no training needed)
- train_agent: Train a PPO agent and collect metrics
- request_improvement: Ask Agent 1 to fix issues

Your workflow:
1. First, analyze the code for obvious issues
2. If critical issues found: request improvement immediately (no need to train)
3. If code looks OK: train briefly to test
4. Analyze training results
5. If quality < 7/10: request improvements and repeat
6. If quality >= 7/10: declare success

Decision logic for first iteration:
- Usually the initial code has issues, so analyze first
- Only train if the code passes basic analysis

Be efficient - don't waste training time on broken code."""


@dataclass
class AgentMessage:
    """A message in the agent conversation."""
    role: str  # "user", "assistant", "tool_result"
    content: str
    tool_calls: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)


class Agent:
    """Base agent with tool-use capabilities."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Dict[str, Callable],
        api_key: str = None,
        model: str = "claude-sonnet-4-5-20250929"
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = ClaudeClient(api_key=api_key)
        self.model = model
        self.messages: List[AgentMessage] = []

    def _format_tools_for_prompt(self) -> str:
        """Format available tools as text for the prompt."""
        tools_desc = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            tools_desc.append(f"- {name}: {doc.strip().split(chr(10))[0]}")
        return "\n".join(tools_desc)

    def _build_prompt(self, user_message: str) -> str:
        """Build the full prompt with system, history, and user message."""
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Available tools:\n{self._format_tools_for_prompt()}\n\n"
        prompt += "To use a tool, respond with:\n"
        prompt += "TOOL: <tool_name>\n"
        prompt += "ARGS: <json args>\n\n"
        prompt += "You can use multiple tools in sequence. After each tool result, decide next action.\n\n"

        # Add conversation history
        for msg in self.messages[-10:]:  # Last 10 messages for context
            if msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
            elif msg.role == "tool_result":
                prompt += f"Tool Result: {msg.content}\n\n"

        prompt += f"User: {user_message}\n\nAssistant:"
        return prompt

    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """Parse a tool call from the response."""
        import re

        # Look for TOOL: and ARGS: pattern
        tool_match = re.search(r'TOOL:\s*(\w+)', response)
        args_match = re.search(r'ARGS:\s*(\{[^}]+\}|\[[^\]]+\]|"[^"]*"|\w+)', response, re.DOTALL)

        if tool_match:
            tool_name = tool_match.group(1)
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except:
                    # Try to extract simple key-value
                    args = {"input": args_match.group(1).strip('"')}
            return {"tool": tool_name, "args": args if isinstance(args, dict) else {"input": args}}

        return None

    def _execute_tool(self, tool_name: str, args: Dict) -> ToolResult:
        """Execute a tool and return the result."""
        if tool_name not in self.tools:
            return ToolResult(False, None, f"Unknown tool: {tool_name}")

        try:
            # Get function and call with args
            func = self.tools[tool_name]

            # Handle different argument styles
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if len(params) == 1 and len(args) == 1:
                # Single argument - pass directly
                result = func(list(args.values())[0])
            else:
                result = func(**args)

            return result
        except Exception as e:
            return ToolResult(False, None, str(e))

    def run(self, user_message: str, max_iterations: int = 10) -> str:
        """Run the agent with a user message, handling tool calls."""
        self.messages.append(AgentMessage(role="user", content=user_message))

        current_message = user_message
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get response from Claude
            prompt = self._build_prompt(current_message)
            response = self.client.call_claude(prompt, model=self.model, max_tokens=4096)

            # Check for tool call
            tool_call = self._parse_tool_call(response)

            if tool_call:
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]

                print(f"\n   [{self.name}] Using tool: {tool_name}")
                print(f"   Args: {json.dumps(tool_args, indent=2)[:200]}...")

                # Execute tool
                result = self._execute_tool(tool_name, tool_args)

                # Format result
                if result.success:
                    result_text = f"Success: {json.dumps(result.data, indent=2, default=str)[:2000]}"
                else:
                    result_text = f"Error: {result.error}"

                print(f"   Result: {'Success' if result.success else 'Error'}")

                # Add to history
                self.messages.append(AgentMessage(role="assistant", content=response))
                self.messages.append(AgentMessage(role="tool_result", content=result_text))

                # Continue with tool result
                current_message = f"Tool result for {tool_name}:\n{result_text}\n\nWhat's next?"
            else:
                # No tool call - final response
                self.messages.append(AgentMessage(role="assistant", content=response))
                return response

        return "Max iterations reached"


# =============================================================================
# Two-Agent Orchestrator
# =============================================================================

class AgentOrchestrator:
    """Coordinates the Generator and Supervisor agents."""

    def __init__(self, api_key: str = None, verbose: bool = True):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.verbose = verbose

        # Create Generator Agent (Agent 1)
        self.generator = Agent(
            name="Generator",
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            tools={
                "browse_game": tool_browse_game,
                "save_environment": tool_save_environment,
            },
            api_key=self.api_key
        )

        # Create Supervisor Agent (Agent 2)
        self.supervisor = Agent(
            name="Supervisor",
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            tools={
                "analyze_code": tool_analyze_code,
                "train_agent": tool_train_agent,
            },
            api_key=self.api_key
        )

        # Shared state
        self.game_info: Optional[Dict] = None
        self.current_code: Optional[str] = None
        self.env_name: Optional[str] = None
        self.iteration = 0
        self.max_iterations = 5

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, game_url: str, hints: str = "") -> Dict[str, Any]:
        """
        Run the full agent pipeline to create a trainable environment.

        1. Generator browses game and creates initial environment
        2. Supervisor analyzes and iteratively improves
        3. Returns final environment info
        """
        self.log(f"\n{'='*60}")
        self.log("UNIWRAP AGENT ORCHESTRATOR")
        self.log(f"{'='*60}")
        self.log(f"Game URL: {game_url}")
        self.log(f"Hints: {hints or 'None'}")

        # Step 1: Generator browses game
        self.log(f"\n[Phase 1] Generator Agent - Analyzing Game")
        self.log("-" * 40)

        browse_result = tool_browse_game(game_url)
        if not browse_result.success:
            return {"success": False, "error": f"Failed to browse game: {browse_result.error}"}

        self.game_info = browse_result.data
        self.log(f"   Title: {self.game_info.get('title', 'Unknown')}")
        self.log(f"   Has Canvas: {self.game_info.get('has_canvas', False)}")
        self.log(f"   Instructions found: {len(self.game_info.get('instructions', []))}")

        # Determine game type
        game_type = "web"
        if "github.com" in game_url and "pygame" in game_url.lower():
            game_type = "pygame"

        # Step 2: Generator creates initial environment
        self.log(f"\n[Phase 2] Generator Agent - Creating Environment")
        self.log("-" * 40)

        gen_result = tool_generate_environment(self.game_info, game_type, hints)
        if not gen_result.success:
            return {"success": False, "error": f"Failed to generate environment: {gen_result.error}"}

        self.current_code = gen_result.data["code"]

        # Create env name from URL
        import re
        url_clean = game_url.replace("https://", "").replace("http://", "")
        self.env_name = re.sub(r'[^a-zA-Z0-9]', '_', url_clean)[:30].strip('_')

        # Save initial version
        save_result = tool_save_environment(self.env_name, self.current_code)
        if save_result.success:
            self.log(f"   Saved: {save_result.data['env_path']}")

        # Step 3: Supervisor loop
        self.log(f"\n[Phase 3] Supervisor Agent - Quality Loop")
        self.log("-" * 40)

        quality_score = 0

        while self.iteration < self.max_iterations and quality_score < 7:
            self.iteration += 1
            self.log(f"\n   --- Iteration {self.iteration}/{self.max_iterations} ---")

            # Analyze code first
            analysis = tool_analyze_code(self.current_code)
            issues = analysis.data["issues"] if analysis.success else []

            self.log(f"   Code Analysis: {len(issues)} issues found")
            for issue in issues[:3]:
                self.log(f"     - [{issue['severity']}] {issue['issue']}")

            # If critical issues, improve without training
            if analysis.data.get("has_critical", False):
                self.log(f"   Critical issues found - requesting improvement (no training)")
                quality_score = max(1, 10 - len(issues) * 2)
            else:
                # Train briefly to test
                self.log(f"   Training agent (5000 steps)...")
                train_result = tool_train_agent(self.env_name, timesteps=5000)

                if train_result.success:
                    data = train_result.data
                    self.log(f"   Training complete: {data['episodes']} episodes")
                    self.log(f"   Avg reward: {data['avg_reward']:.2f}, Avg length: {data['avg_length']:.0f}")

                    # Calculate quality score
                    # Good: reward improving, episodes > 50 steps, reward variance
                    quality_score = 5  # Base
                    if data['avg_length'] > 50:
                        quality_score += 1
                    if data['avg_length'] > 100:
                        quality_score += 1
                    if data['reward_std'] > 0.1:
                        quality_score += 1
                    if data['avg_reward'] > 0:
                        quality_score += 1
                    if len(issues) == 0:
                        quality_score += 1

                    quality_score = min(10, quality_score)
                    self.log(f"   Quality Score: {quality_score}/10")
                else:
                    self.log(f"   Training failed: {train_result.error[:100]}")
                    quality_score = 2

            # If quality not good enough, improve
            if quality_score < 7 and self.iteration < self.max_iterations:
                self.log(f"   Requesting improvements from Generator...")

                improve_result = tool_improve_environment(
                    self.current_code,
                    issues,
                    self.game_info
                )

                if improve_result.success:
                    self.current_code = improve_result.data["improved_code"]

                    # Save new version
                    new_env_name = f"{self.env_name}_v{self.iteration + 1}"
                    save_result = tool_save_environment(new_env_name, self.current_code)
                    if save_result.success:
                        self.env_name = new_env_name
                        self.log(f"   Saved improved version: {new_env_name}")
                        self.log(f"   Changelog: {improve_result.data.get('changelog', '')[:200]}")
                else:
                    self.log(f"   Improvement failed: {improve_result.error}")

        # Final result
        self.log(f"\n{'='*60}")
        self.log("ORCHESTRATION COMPLETE")
        self.log(f"{'='*60}")
        self.log(f"Final Environment: {self.env_name}")
        self.log(f"Quality Score: {quality_score}/10")
        self.log(f"Iterations: {self.iteration}")

        return {
            "success": True,
            "env_name": self.env_name,
            "quality_score": quality_score,
            "iterations": self.iteration,
            "game_info": self.game_info
        }


def run_agents(game_url: str, hints: str = "") -> Dict[str, Any]:
    """Convenience function to run the agent orchestrator."""
    orchestrator = AgentOrchestrator()
    return orchestrator.run(game_url, hints)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m uniwrap.agents <game_url> [hints]")
        sys.exit(1)

    url = sys.argv[1]
    hints = sys.argv[2] if len(sys.argv) > 2 else ""

    result = run_agents(url, hints)
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
