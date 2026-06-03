from __future__ import annotations

from agent_watermark.agent_core.langgraph_agent import AgentConfig, WatermarkedLangGraphAgent
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import WatermarkIdentity


class WatermarkedLangChainAgent(WatermarkedLangGraphAgent):
    """Compatibility runner for platform migration experiments.

    The planner uses LangChain ChatOpenAI and LangChain BaseTool objects, while the
    control loop is a plain ReAct loop instead of a LangGraph StateGraph.
    """

    def __init__(self, tools, logger: JsonlExecutionLogger, identity: WatermarkIdentity, config: AgentConfig):
        super().__init__(tools, logger, identity, config)

    def run(self, task: str):
        state = {"task": task, "scratchpad": [], "step_index": 0, "final_answer": None}
        for _ in range(self.config.max_steps):
            state = self._planner(state)
            if state["scratchpad"][-1]["chosen_action"] == "final_answer":
                self._log_step(state, None)
                break
            state = self._tool_executor(state)
        return {"run_id": self.run_id, "answer": state.get("final_answer"), "scratchpad": state["scratchpad"]}
