from __future__ import annotations

import time
import logging
from typing import Any

from core.types import (
    Role,
    Message,
    ToolCall,
    ToolResult,
    Task,
    StepResult,
    TerminationReason,
    EvalResult,
)
from typing import TYPE_CHECKING
from core.base import AgentAdapter, Environment, Evaluator
if TYPE_CHECKING:
    from core.base import UserSimulator as UserSimulatorType

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the Agent <-> Environment interaction loop and collects evaluation results."""

    def __init__(
        self,
        agent: AgentAdapter,
        env: Environment,
        evaluators: dict[str, Evaluator],
        user: UserSimulatorType | None = None,
    ):
        self.agent = agent
        self.env = env
        self.evaluators = evaluators
        self.user = user

    def run(self, task: Task) -> EvalResult:
        start = time.time()
        self.agent.reset()
        if self.user:
            self.user.reset(task)
        init_obs = self.env.reset(task)

        # Build initial conversation
        trajectory: list[Message] = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt(init_obs, task)),
            Message(role=Role.USER, content=task.initial_message),
        ]

        termination = TerminationReason.MAX_STEPS
        steps = 0
        step_durations: list[float] = []
        step_rewards: list[float] = []
        # Get state_match evaluator for progress tracking (if available)
        _progress_evaluator = self.evaluators.get("state_match")

        for step in range(task.max_steps):
            steps = step + 1
            step_start = time.time()

            # ── Agent generates a response ──
            try:
                agent_msg = self.agent.act(trajectory)
            except Exception as e:
                logger.error("Agent error at step %d: %s", step, e)
                step_durations.append(time.time() - step_start)
                termination = TerminationReason.AGENT_ERROR
                break

            trajectory.append(agent_msg)

            # ── Case 1: text-only reply (no tool calls) ──
            if not agent_msg.tool_calls:
                if self._is_stop_signal(agent_msg) or task.single_turn:
                    step_durations.append(time.time() - step_start)
                    termination = TerminationReason.SUCCESS
                    break

                # Route to user simulator if available
                if self.user:
                    user_msg = self.user.respond(task, trajectory)
                    if user_msg is None:
                        step_durations.append(time.time() - step_start)
                        termination = TerminationReason.USER_STOP
                        break
                    trajectory.append(user_msg)
                step_durations.append(time.time() - step_start)
                continue

            # ── Case 2: tool calls ──
            tool_results: list[ToolResult] = []
            task_done = False

            for tc in agent_msg.tool_calls:
                try:
                    result = self.env.step(tc)
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        output=result.observation,
                        error=False,
                    ))
                    if result.done:
                        task_done = True
                except Exception as e:
                    logger.error("Env error on tool %s: %s", tc.name, e)
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        output=f"Error: {e}",
                        error=True,
                    ))

            trajectory.append(Message(role=Role.ENV, tool_results=tool_results))

            # Track intermediate reward for progress rate
            if _progress_evaluator:
                try:
                    reward = _progress_evaluator.evaluate(task, trajectory, self.env)
                    step_rewards.append(reward)
                except Exception:
                    step_rewards.append(step_rewards[-1] if step_rewards else 0.0)

            step_durations.append(time.time() - step_start)

            if task_done:
                termination = TerminationReason.SUCCESS
                break

        # ── Evaluation ──
        scores: dict[str, float] = {}
        for name, evaluator in self.evaluators.items():
            try:
                scores[name] = evaluator.evaluate(task, trajectory, self.env)
            except Exception as e:
                logger.error("Evaluator '%s' failed: %s", name, e)
                scores[name] = 0.0

        overall = 1.0
        for s in scores.values():
            overall *= s

        # Compute progress rate: area under step-reward curve / max possible
        progress_rate = 0.0
        if step_rewards:
            progress_rate = sum(step_rewards) / len(step_rewards)

        elapsed = time.time() - start
        logger.info(
            "Task %s done in %.1fs | %d steps | %s | overall=%.2f | progress=%.2f",
            task.id, elapsed, steps, termination.value, overall, progress_rate,
        )

        return EvalResult(
            task_id=task.id,
            terminated=termination,
            trajectory=trajectory,
            scores=scores,
            overall_score=overall,
            steps_taken=steps,
            elapsed_seconds=elapsed,
            step_durations=step_durations,
            step_rewards=step_rewards,
            progress_rate=progress_rate,
        )

    def _build_system_prompt(self, init_obs: str, task: Task) -> str:
        # HTTP Agent (passthrough): no tools, no "done" instruction
        if not self.env.get_tool_schemas():
            return f"## 任务描述\n{task.description}"

        # Tool-calling Agent: full prompt with tool instructions
        return (
            "你是一个专业的客服助手。"
            "请使用提供的工具来完成用户的请求。"
            "任务全部完成后，请调用 'done' 工具。\n\n"
            f"## 环境信息\n{init_obs}\n\n"
            f"## 任务描述\n{task.description}"
        )

    @staticmethod
    def _is_stop_signal(msg: Message) -> bool:
        if msg.content and "###DONE###" in msg.content:
            return True
        return False
