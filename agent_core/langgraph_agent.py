from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.logging.schemas import StepLog, ToolCallLog, utc_now_iso
from agent_watermark.watermark.embedder import MultiStatisticWatermarkEmbedder
from agent_watermark.watermark.signature import WatermarkIdentity


class AgentState(TypedDict):
    task: str
    scratchpad: List[Dict[str, Any]]
    step_index: int
    final_answer: str | None


@dataclass
class AgentConfig:
    model: str
    temperature: float
    max_steps: int
    watermark_lambda: float


class WatermarkedLangGraphAgent:
    """LangGraph ReAct-style agent with a watermarked action-selection middleware."""

    def __init__(
        self,
        tools: List[BaseTool],
        logger: JsonlExecutionLogger,
        identity: WatermarkIdentity,
        config: AgentConfig,
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.logger = logger
        self.identity = identity
        self.config = config
        self.llm = ChatOpenAI(model=config.model, temperature=config.temperature)
        self.embedder = MultiStatisticWatermarkEmbedder(identity, config.watermark_lambda)
        self.run_id = str(uuid.uuid4())
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner)
        graph.add_node("tool_executor", self._tool_executor)
        graph.set_entry_point("planner")
        graph.add_conditional_edges("planner", self._route_after_planner, {"tool": "tool_executor", "end": END})
        graph.add_edge("tool_executor", "planner")
        return graph.compile()

    def _candidate_descriptions(self) -> Dict[str, str]:
        desc = {name: tool.description for name, tool in self.tools.items()}
        desc["final_answer"] = "Finish the task and answer the user."
        return desc

    def _planner_prompt(self, state: AgentState) -> List[Any]:
        tool_docs = "\n".join(f"- {name}: {tool.description}" for name, tool in self.tools.items())
        schema = {
            "reasoning": "short private rationale",
            "candidate_logits": {"search": 0.0, "sqlite_db": 0.0, "python_repl": 0.0, "file_system": 0.0, "final_answer": 0.0},
            "arguments": {"query": "or other tool args"},
            "final_answer": "only when action is final_answer",
        }
        return [
            SystemMessage(
                content=(
                    "You are a ReAct tool-using research agent. Select the next action from the listed tools "
                    "or final_answer. Return valid JSON only. Candidate logits must score every action by "
                    "task utility before any watermarking. Do not mention watermarking.\n"
                    f"Tools:\n{tool_docs}"
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {"task": state["task"], "scratchpad": state["scratchpad"], "response_schema": schema},
                    ensure_ascii=False,
                )
            ),
        ]

    def _parse_json(self, content: str) -> Dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1)
        return json.loads(text)

    def _planner(self, state: AgentState) -> AgentState:
        response = self.llm.invoke(self._planner_prompt(state))
        parsed = self._parse_json(str(response.content))
        descriptions = self._candidate_descriptions()
        raw_logits = {name: float(parsed.get("candidate_logits", {}).get(name, -5.0)) for name in descriptions}
        candidates = self.embedder.reweight(raw_logits, descriptions)
        chosen = self.embedder.choose(candidates)
        arguments = parsed.get("arguments", {}) if chosen.name != "final_answer" else {}
        final_answer = parsed.get("final_answer") if chosen.name == "final_answer" else None
        step = {
            "reasoning": parsed.get("reasoning", ""),
            "candidate_actions": [c.model_dump() for c in candidates],
            "chosen_action": chosen.name,
            "arguments": arguments,
            "final_answer": final_answer,
        }
        return {**state, "scratchpad": state["scratchpad"] + [step], "final_answer": final_answer, "step_index": state["step_index"] + 1}

    def _route_after_planner(self, state: AgentState) -> str:
        last = state["scratchpad"][-1]
        if last["chosen_action"] == "final_answer" or state["step_index"] >= self.config.max_steps:
            return "end"
        return "tool"

    def _tool_executor(self, state: AgentState) -> AgentState:
        last = state["scratchpad"][-1]
        name = last["chosen_action"]
        call = ToolCallLog(tool_name=name, arguments=last.get("arguments", {}))
        try:
            observation = self.tools[name].invoke(last.get("arguments", {}))
            call.observation = observation
        except Exception as exc:
            call.error = repr(exc)
            observation = f"ERROR: {exc!r}"
        call.finished_at = utc_now_iso()
        last["observation"] = observation
        last["tool_call"] = call.model_dump()
        self._log_step(state, call)
        return state

    def _log_step(self, state: AgentState, call: ToolCallLog | None = None) -> None:
        last = state["scratchpad"][-1]
        self.logger.append(
            StepLog(
                run_id=self.run_id,
                step_index=state["step_index"] - 1,
                author_id=self.identity.author_id,
                watermark_timestamp=self.identity.timestamp,
                task=state["task"],
                state={"scratchpad_size": len(state["scratchpad"])},
                reasoning_trace=last.get("reasoning", ""),
                candidate_actions=last["candidate_actions"],
                chosen_action=last["chosen_action"],
                chosen_arguments=last.get("arguments", {}),
                tool_call=call,
                final_answer=last.get("final_answer"),
            )
        )

    def run(self, task: str) -> Dict[str, Any]:
        initial: AgentState = {"task": task, "scratchpad": [], "step_index": 0, "final_answer": None}
        final_state = self.graph.invoke(initial)
        if final_state["scratchpad"] and final_state["scratchpad"][-1]["chosen_action"] == "final_answer":
            self._log_step(final_state, None)
        return {"run_id": self.run_id, "answer": final_state.get("final_answer"), "scratchpad": final_state["scratchpad"]}
