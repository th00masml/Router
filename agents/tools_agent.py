from __future__ import annotations

from typing import List, Dict, Any

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

from tools.basic_tools import get_basic_tools


def build_tools_agent(llm, *, max_iterations: int = 10, verbose: bool = False, tools_override=None, strict: bool = False) -> AgentExecutor:
    tools = tools_override or get_basic_tools()
    effective_max = 2 if strict else max_iterations
    sys_text = (
        "You are a careful AI assistant. Use tools when they help you answer accurately.\n\n"
        "Available tools:\n{tools}\n\n"
        "Valid tool names: {tool_names}.\n\n"
        "Follow this exact interaction format when reasoning and taking actions:\n"
        "Thought: describe your reasoning.\n"
        "Action: the tool to use, must be EXACTLY one of [{tool_names}] (no other names)\n"
        "Action Input: the input for the tool (plain text). For 'now' and 'today', leave the input empty.\n"
        "Observation: the result of the tool\n"
        "(You may repeat Thought/Action/Action Input/Observation up to {max_iter} times.)\n"
        "When you can answer, finish with:\n"
        "Thought: I now know the final answer\n"
        "Final Answer: your concise answer (cite URLs you fetched if applicable).\n\n"
        "Rules:\n"
        "- After writing 'Thought:' you MUST write either 'Action:' (with 'Action Input:') or 'Final Answer:' next.\n"
        "- Never write two 'Thought:' lines in a row.\n"
        "- Keep steps minimal (≤ {max_iter}).\n"
        "- For queries about current date/time, call 'now' or 'today' first (do not browse).\n"
        "- Do not invent tools or leave the tool name blank. If no tool applies, proceed to Final Answer.\n"
        "- Do not call the same tool more than once unless new input is needed. For 'now'/'today', call at most once.\n"
    ).replace("{max_iter}", str(effective_max))
    # A couple of few-shot examples to enforce the format and preferred tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_text),
        # Few-shot 1: current date
        ("human", "What is today's date?"),
        ("ai", "Thought: I should use the date tool.\nAction: today\nAction Input: \nObservation: 2025-09-01\nThought: I now know the final answer\nFinal Answer: Today is 2025-09-01."),
        # Few-shot 2: current time in timezone
        ("human", "What time is it now in Europe/Warsaw?"),
        ("ai", "Thought: I should get the time for a specific timezone.\nAction: time_in\nAction Input: Europe/Warsaw\nObservation: {{\"tz\":\"Europe/Warsaw\",\"iso\":\"2025-09-01T12:34:56+02:00\",\"time\":\"12:34:56\"}}\nThought: I now know the final answer\nFinal Answer: It is about 12:34 in Europe/Warsaw (example)."),
        # Few-shot 3: summarize a URL directly
        ("human", "Summarize https://example.com in 2 sentences."),
        ("ai", "Thought: I should fetch the page directly.\nAction: web_get\nAction Input: https://example.com\nObservation: Example Domain. This domain is for use in illustrative examples...\nThought: I now know the final answer\nFinal Answer: It is a placeholder page explaining the example domain and linking to more information."),
        # Actual input slot
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=verbose,
        max_iterations=effective_max,
        return_intermediate_steps=True,
    )
    return executor


def invoke_tools_agent(executor: AgentExecutor, chat_history: List[AnyMessage], user_input: str, run_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # For broad compatibility, pass only `input`; rely on agent scratchpad.
    if run_config is None:
        run_config = {}
    res: Dict[str, Any] = executor.invoke({"input": user_input}, config=run_config)  # type: ignore[assignment]
    steps = []
    raw_steps = res.get("intermediate_steps") or []
    for s in raw_steps:
        try:
            action, observation = s
            tool = getattr(action, "tool", "")
            tool_input = getattr(action, "tool_input", "")
            steps.append({
                "tool": str(tool),
                "input": str(tool_input),
                "observation": str(observation),
            })
        except Exception:
            steps.append({"raw": str(s)})
    return {"output": res.get("output", ""), "steps": steps}
