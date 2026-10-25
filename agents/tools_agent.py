"""ReAct agent implementation with tool support.

This module provides a configurable ReAct agent that can use various tools
to answer user queries. The agent follows a Thought-Action-Observation loop
until it reaches a final answer.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from tools.basic_tools import get_basic_tools


# System prompt template for the ReAct agent
SYSTEM_PROMPT_TEMPLATE = """You are a careful AI assistant. Use tools when they help you answer accurately.

Available tools:
{tools}

Valid tool names: {tool_names}.

Follow this exact interaction format when reasoning and taking actions:
Thought: describe your reasoning.
Action: the tool to use, must be EXACTLY one of [{tool_names}] (no other names)
Action Input: the input for the tool (plain text). For 'now' and 'today', leave the input empty.
Observation: the result of the tool
(You may repeat Thought/Action/Action Input/Observation up to {max_iter} times.)
When you can answer, finish with:
Thought: I now know the final answer
Final Answer: your concise answer (cite URLs you fetched if applicable).

Rules:
- After writing 'Thought:' you MUST write either 'Action:' (with 'Action Input:') or 'Final Answer:' next.
- Never write two 'Thought:' lines in a row.
- Keep steps minimal (≤ {max_iter}).
- For queries about current date/time, call 'now' or 'today' first (do not browse).
- Do not invent tools or leave the tool name blank. If no tool applies, proceed to Final Answer.
- Do not call the same tool more than once unless new input is needed. For 'now'/'today', call at most once.
"""


# Few-shot examples to guide the agent's behavior
FEW_SHOT_EXAMPLES = [
    # Example 1: Current date query
    (
        "What is today's date?",
        "Thought: I should use the date tool.\n"
        "Action: today\n"
        "Action Input: \n"
        "Observation: 2025-09-01\n"
        "Thought: I now know the final answer\n"
        "Final Answer: Today is 2025-09-01."
    ),
    # Example 2: Timezone query
    (
        "What time is it now in Europe/Warsaw?",
        "Thought: I should get the time for a specific timezone.\n"
        "Action: time_in\n"
        "Action Input: Europe/Warsaw\n"
        'Observation: {"tz":"Europe/Warsaw","iso":"2025-09-01T12:34:56+02:00","time":"12:34:56"}\n'
        "Thought: I now know the final answer\n"
        "Final Answer: It is about 12:34 in Europe/Warsaw (example)."
    ),
    # Example 3: URL summarization
    (
        "Summarize https://example.com in 2 sentences.",
        "Thought: I should fetch the page directly.\n"
        "Action: web_get\n"
        "Action Input: https://example.com\n"
        "Observation: Example Domain. This domain is for use in illustrative examples...\n"
        "Thought: I now know the final answer\n"
        "Final Answer: It is a placeholder page explaining the example domain and linking to more information."
    ),
]


def _build_prompt(max_iterations: int) -> ChatPromptTemplate:
    """Build the prompt template for the ReAct agent.

    Args:
        max_iterations: Maximum number of tool invocation iterations

    Returns:
        Configured ChatPromptTemplate with system prompt and few-shot examples
    """
    # Format system prompt with iteration limit
    system_text = SYSTEM_PROMPT_TEMPLATE.replace("{max_iter}", str(max_iterations))

    # Build message list: system prompt + few-shot examples + input slots
    messages = [("system", system_text)]

    # Add few-shot examples
    for human_msg, ai_msg in FEW_SHOT_EXAMPLES:
        messages.append(("human", human_msg))
        messages.append(("ai", ai_msg))

    # Add actual input slots
    messages.extend([
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])

    return ChatPromptTemplate.from_messages(messages)


def _process_intermediate_steps(raw_steps: List[Tuple[Any, Any]]) -> List[Dict[str, str]]:
    """Process raw agent steps into a clean format.

    Args:
        raw_steps: List of (action, observation) tuples from agent execution

    Returns:
        List of dicts with tool, input, and observation fields
    """
    steps = []
    for step in raw_steps:
        try:
            action, observation = step
            tool = getattr(action, "tool", "")
            tool_input = getattr(action, "tool_input", "")
            steps.append({
                "tool": str(tool),
                "input": str(tool_input),
                "observation": str(observation),
            })
        except Exception:
            # Fallback for unparseable steps
            steps.append({"raw": str(step)})
    return steps


def build_tools_agent(
    llm,
    *,
    max_iterations: int = 10,
    verbose: bool = False,
    tools_override: Optional[List] = None,
    strict: bool = False
) -> AgentExecutor:
    """Build a ReAct agent executor with the specified configuration.

    Args:
        llm: Language model to use for agent reasoning
        max_iterations: Maximum number of tool invocation iterations (default: 10)
        verbose: Enable verbose logging of agent steps (default: False)
        tools_override: Optional list of tools to use instead of defaults
        strict: Enable strict mode with reduced iterations for time-sensitive queries (default: False)

    Returns:
        Configured AgentExecutor ready for invocation
    """
    # Get tools (use override if provided, otherwise defaults)
    tools = tools_override or get_basic_tools()

    # Apply strict mode: reduce iterations for fast queries
    effective_max_iterations = 2 if strict else max_iterations

    # Build prompt template
    prompt = _build_prompt(effective_max_iterations)

    # Create ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Wrap in executor with error handling
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=verbose,
        max_iterations=effective_max_iterations,
        return_intermediate_steps=True,
    )

    return executor


def invoke_tools_agent(
    executor: AgentExecutor,
    chat_history: List[AnyMessage],
    user_input: str,
    run_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Invoke the tools agent and return formatted results.

    Args:
        executor: Configured AgentExecutor instance
        chat_history: Previous conversation messages (currently unused, for future context support)
        user_input: User's current input/query
        run_config: Optional LangChain run configuration (e.g., for LangSmith tracing)

    Returns:
        Dict with 'output' (final answer) and 'steps' (list of tool invocations)
    """
    if run_config is None:
        run_config = {}

    # Invoke agent with user input
    result: Dict[str, Any] = executor.invoke(
        {"input": user_input},
        config=run_config
    )  # type: ignore[assignment]

    # Extract and format intermediate steps
    raw_steps = result.get("intermediate_steps") or []
    formatted_steps = _process_intermediate_steps(raw_steps)

    return {
        "output": result.get("output", ""),
        "steps": formatted_steps
    }
