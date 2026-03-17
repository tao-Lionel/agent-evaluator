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
from core.base import AgentAdapter, Environment, Evaluator

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the Agent <-> Environment interaction loop and collects evaluation results."""

    def __init__(
        self,
        agent: AgentAdapter,
        env: Environment,
        evaluators: dict[str, Evaluator],
    ):
        self.agent = agent
        self.env = env
        self.evaluators = evaluators

    def run(self, task: Task) -> EvalResult:
        start = time.time()
        self.agent.reset()
        init_obs = self.env.reset(task)

        # Build initial conversation
        trajectory: list[Message] = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt(init_obs, task)),
            Message(role=Role.USER, content=task.initial_message),
        ]

        termination = TerminationReason.MAX_STEPS
        steps = 0

        for step in range(task.max_steps):
            steps = step + 1

            # ── Agent generates a response ──
            try:
                agent_msg = self.agent.act(trajectory)
            except Exception as e:
                logger.error("Agent error at step %d: %s", step, e)
                termination = TerminationReason.AGENT_ERROR
                break

            trajectory.append(agent_msg)

            # ── Case 1: text-only reply (no tool calls) ──
            if not agent_msg.tool_calls:
                if self._is_stop_signal(agent_msg):
                    termination = TerminationReason.SUCCESS
                    break
                # Agent said something but didn't call tools — next iteration
                # will feed this back in; in MVP we just continue
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

        elapsed = time.time() - start
        logger.info(
            "Task %s done in %.1fs | %d steps | %s | overall=%.2f",
            task.id, elapsed, steps, termination.value, overall,
        )

        return EvalResult(
            task_id=task.id,
            terminated=termination,
            trajectory=trajectory,
            scores=scores,
            overall_score=overall,
            steps_taken=steps,
        )

    @staticmethod
    def _build_system_prompt(init_obs: str, task: Task) -> str:
        return (
            "You are a helpful customer service agent. "
            "Use the provided tools to fulfill the user's request. "
            "When the task is fully completed, call the 'done' tool.\n\n"
            f"## Environment Info\n{init_obs}\n\n"
            f"## Task Description\n{task.description}"
        )

    @staticmethod
    def _is_stop_signal(msg: Message) -> bool:
        if msg.content and "###DONE###" in msg.content:
            return True
        return False
