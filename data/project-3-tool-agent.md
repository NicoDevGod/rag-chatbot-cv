# Project 3: Tool-use agent

Tool-use agent: Nicolas's third AI project is an agent that decides when to call real tools: a calculator, weather, Wikipedia and the current time. It is built without agent frameworks, directly on Groq's native function-calling API, so the whole protocol is visible. Code: https://github.com/NicoDevGod/tool-agent. Live demo: https://tool-agent-jgfi.onrender.com.

Tool-use agent, why: An LLM only generates text. It cannot calculate precisely and does not know today's weather or the time, and if asked it invents plausible but false answers. With function calling the model can instead request something like "run calculator with expression 23*4"; Python executes the real function and returns the result, and the model finishes its answer with real data.

Tool-use agent, the loop: Each tool is described with a JSON schema (name, purpose, arguments). The code calls the model with the tool schemas; if the model asks for a tool, Python runs it, appends the result with role "tool" and calls the model again; if not, that reply is the final answer. A MAX_STEPS limit of 5 stops buggy agents from calling tools forever. The Gradio interface shows each tool call before the answer. The model is openai/gpt-oss-20b on Groq.

Tool-use agent, the four tools: calculator (evaluates arithmetic safely by parsing the expression with Python's ast module and allowing only numbers and operators, never raw eval(), because the LLM writes that string), get_weather (Open-Meteo geocoding plus current forecast, no API key), search_wikipedia (first searches Wikipedia to find the real page title, then fetches its summary) and get_current_time (standard-library zoneinfo, any IANA timezone).

Tool-use agent, problems solved: (1) Groq rejected the assistant message copied with message.model_dump() because it contained extra fields such as annotations; he builds the message by hand with only role, content and tool_calls. (2) Wikipedia returned HTTP 403 without a User-Agent header, so a descriptive one was added. (3) When a search failed, the agent kept retrying reworded queries until it used all 5 steps; the fix was resolving the real title first and adding a prompt rule to not retry and to answer from its own knowledge.

Tool-use agent, verified result: In production a combined question about the weather in Ovalle plus a calculation made the agent call get_weather correctly (about 13 degrees Celsius) and solve both parts.
