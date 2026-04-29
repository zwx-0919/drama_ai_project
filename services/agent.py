from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from time import perf_counter
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from services.llm import ScriptEngine
from services.memory import RedisChatMessageHistory
from services.rag import RAGService


@dataclass
class AgentStep:
    action: str
    reason: str
    tool: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


class ReActDramaAgent:
    STYLE_KEYWORDS = ["甜宠", "爽文", "悬疑", "古风", "逆袭", "校园", "职场", "现实", "轻喜"]
    TOPIC_KEYWORDS = ["校园", "都市", "豪门", "重生", "逆袭", "反转", "家庭", "爱情", "悬疑", "职场"]

    def __init__(self, engine: ScriptEngine, rag: RAGService, memory_service: RedisChatMessageHistory) -> None:
        self.engine = engine
        self.rag = rag
        self.memory_service = memory_service

    def _classify_goal(self, goal: str) -> str:
        text = goal.lower()
        if any(keyword in text for keyword in ["相似", "仿写", "参考", "同款风格", "同结构", "同节奏", "类似剧情"]):
            return "similar"
        if any(keyword in text for keyword in ["检索", "搜索", "查找"]):
            return "search"
        if any(keyword in text for keyword in ["改写", "优化", "润色", "重写"]):
            return "rewrite"
        if any(keyword in text for keyword in ["续写", "继续"]):
            return "continue"
        if any(keyword in text for keyword in ["扩写", "补充", "丰富"]):
            return "expand"
        if any(keyword in text for keyword in ["压缩", "精简", "缩短"]):
            return "shorten"
        if any(keyword in text for keyword in ["分镜", "镜头"]):
            return "shot"
        if any(keyword in text for keyword in ["元信息", "标题", "关键词", "角色", "评分"]):
            return "meta"
        if any(keyword in text for keyword in ["生成", "创作", "写一个", "写出", "帮我写", "短剧", "剧本"]):
            return "auto_generate"
        return "general"

    def _extract_generation_params(self, goal: str) -> dict[str, Any]:
        text = goal.replace("，", " ").replace("。", " ")
        duration_min = 1
        minute_match = re.search(r"(\d+)\s*分钟", text)
        second_match = re.search(r"(\d+)\s*秒", text)
        if minute_match:
            duration_min = max(1, int(minute_match.group(1)))
        elif second_match:
            seconds = int(second_match.group(1))
            duration_min = max(1, round(seconds / 60))

        style = next((item for item in self.STYLE_KEYWORDS if item in text), "爽文")
        keywords = [item for item in self.TOPIC_KEYWORDS if item in text]
        if not keywords:
            keywords = [token for token in re.split(r"[\s,，、]+", text) if 1 < len(token) <= 6][:6]

        theme_parts = [item for item in ["校园", "重生", "逆袭", "大女主", "悬疑", "爱情", "豪门", "家庭", "职场"] if item in text]
        theme = "".join(theme_parts[:3]) or "短剧"
        return {"theme": theme, "style": style, "duration_min": duration_min, "keywords": keywords[:8]}

    def _normalize_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        return [item for item in history if item.get("role") in {"user", "assistant"} and item.get("content")]

    def _recent_history(self, history: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
        return history[-limit:] if history else []

    def _history_summary(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "无历史摘要"
        summary = self.engine.compress_memory(history)
        parts = []
        if summary.get("worldview"):
            parts.append(f"世界观/人设：{summary['worldview']}")
        if summary.get("character_state"):
            parts.append(f"人物状态：{summary['character_state']}")
        if summary.get("plot_progress"):
            parts.append(f"剧情进度：{summary['plot_progress']}")
        if summary.get("pending_tasks"):
            parts.append(f"待办事项：{summary['pending_tasks']}")
        return "；".join(parts) or "无历史摘要"

    def _last_assistant_script(self, history: list[dict[str, str]]) -> str:
        for item in reversed(history):
            if item.get("role") == "assistant" and item.get("content"):
                return item["content"]
        return ""

    def _observe_search(self, user_id: str, brief: str, top_k: int, drama_id: str | None = None) -> tuple[list[dict[str, Any]], AgentStep]:
        start = perf_counter()
        results = self.rag.search(user_id, drama_id or brief, brief, top_k=top_k)
        return results, AgentStep(action="observe", reason="检索与目标最相关的历史内容，作为生成上下文。", tool="rag.search", input={"user_id": user_id, "query": brief, "top_k": top_k, "drama_id": drama_id}, output={"results": results}, latency_ms=(perf_counter() - start) * 1000)

    def _build_context(self, user_id: str, brief: str, top_k: int, drama_id: str | None = None) -> tuple[str, list[AgentStep], list[dict[str, str]], list[dict[str, Any]], str]:
        history = self._normalize_history(self.memory_service.get_messages(user_id, project_id=drama_id or "project_001"))
        recent_history = self._recent_history(history, limit=3)
        recent_docs = self.rag.get_recent_documents(user_id, drama_id=drama_id, limit=5)
        summary = self.memory_service.get_summary(user_id, project_id=drama_id or "project_001")
        summary_text = self._history_summary(summary)
        search_results, search_step = self._observe_search(user_id, brief, top_k, drama_id=drama_id)
        context_parts = [brief]
        if summary_text and summary_text != "无历史摘要":
            context_parts.insert(0, f"[历史摘要] {summary_text}")
        if recent_docs:
            context_parts.insert(0, "\n".join(f"[上传文件] {item.get('doc_id')}: {item.get('preview', '')}" for item in recent_docs))
        if recent_history:
            context_parts.extend([item["content"] for item in recent_history])
        if search_results:
            context_parts.insert(0, search_results[0].get("content", ""))
        return "\n\n".join(part for part in context_parts if part), [search_step], history, recent_docs, summary_text

    def _thinking(self, step: AgentStep) -> str:
        return f"{step.action}: {step.reason}"

    def _make_step(self, action: str, reason: str, tool: str, input_data: dict[str, Any], output_data: dict[str, Any], started_at: float) -> AgentStep:
        return AgentStep(action=action, reason=reason, tool=tool, input=input_data, output=output_data, latency_ms=(perf_counter() - started_at) * 1000)

    def _format_output(self, intent: str, steps: list[AgentStep], final_result: dict[str, Any], context_used: str, started_at: float | None = None) -> dict[str, Any]:
        content = final_result.get("content")
        if content is None and intent == "search":
            content = "已完成检索，请查看 results。"
        thinking_steps = [
            {
                "step": idx,
                "action": step.action,
                "reason": step.reason,
                "tool": step.tool,
                "input": step.input,
                "output_summary": list(step.output.keys()) if isinstance(step.output, dict) else [],
                "latency_ms": round(step.latency_ms, 2) if step.latency_ms is not None else None,
            }
            for idx, step in enumerate(steps, start=1)
        ]
        total_latency_ms = None if started_at is None else round((perf_counter() - started_at) * 1000, 2)
        return {"thinking": [self._thinking(step) for step in steps], "thinking_steps": thinking_steps, "intent": intent, "context_used": context_used, "tool_used": [step.tool for step in steps if step.tool], "content": content or "", "final_result": final_result, "total_latency_ms": total_latency_ms, "steps": [step.__dict__ for step in steps]}

    def _build_react_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.engine.model_name, api_key=self.engine.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0.2)

    def _build_react_prompt(self) -> PromptTemplate:
        template = """你是一个专业短剧编剧与中文助手，必须使用工具完成任务。

可用工具：
{tools}

上下文说明：
{context_meta}

严格遵守：
- 需要处理剧本生成、改写、续写、扩写、精简、分镜、相似剧本、元信息、检索、通用回复时，优先使用最合适的工具。
- 如果上下文里已经给出剧本或检索结果，请结合上下文作答，不要编造。
- 如果你不确定事实，请说明不确定。
- 必须采用 ReAct 格式：
Thought: ...
Action: <tool name>
Action Input: <input>
Observation: <tool result>
... 
Final: <final answer>

上下文：
{context}

问题：
{input}

{agent_scratchpad}"""
        return PromptTemplate.from_template(template)

    def _safe_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _coerce_json(self, raw_input: str) -> dict[str, Any]:
        if not raw_input:
            return {}
        if raw_input.strip().startswith("{"):
            try:
                return json.loads(raw_input)
            except Exception:
                return {"input": raw_input}
        return {"input": raw_input}

    def _build_tools(self, user_id: str, context: str, top_k: int, drama_id: str | None = None) -> dict[str, Tool]:
        def generate_script_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            params = data.get("params") or self._extract_generation_params(data.get("goal", raw_input))
            return self.engine.generate_script(params["theme"], params["style"], params["duration_min"], params["keywords"])

        def rewrite_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            return self.engine.rewrite_script(data.get("content", context), data.get("instruction", "根据要求改写剧本"))

        def continue_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            return self.engine.continue_script(data.get("content", context))

        def expand_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            return self.engine.expand_script(data.get("content", context))

        def shorten_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            return self.engine.shorten_script(data.get("content", context))

        def shot_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            return self.engine.generate_shot_script(data.get("content", context))

        def similar_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            instruction = data.get("instruction", raw_input)
            reference = data.get("reference") or self.rag.get_retriever_context(user_id, drama_id or instruction, instruction, top_k=top_k)
            return self.engine.generate_similar_script(instruction, reference or context)

        def meta_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            content = data.get("content", context)
            meta = {"title": self.engine.generate_title(content), "keywords": self.engine.extract_keywords(content), "characters": self.engine.generate_characters(content), "score": self.engine.score_script(content)}
            return self._safe_json(meta)

        def search_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            results = self.rag.search(user_id, drama_id or data.get("query", raw_input), data.get("query", raw_input), top_k=top_k)
            return self._safe_json(results)

        def general_tool(raw_input: str) -> str:
            data = self._coerce_json(raw_input)
            return self.engine.generate_reply(data.get("message", raw_input), self.memory_service.get_messages(user_id, project_id=drama_id or "project_001"), extra_context=context)

        tools = {
            "generate_script": Tool(name="generate_script", func=generate_script_tool, description="生成完整短剧剧本。输入 JSON: {goal, params}."),
            "rewrite_script": Tool(name="rewrite_script", func=rewrite_tool, description="改写剧本。输入 JSON: {content, instruction}."),
            "continue_script": Tool(name="continue_script", func=continue_tool, description="续写剧本。输入 JSON: {content}."),
            "expand_script": Tool(name="expand_script", func=expand_tool, description="扩写剧本。输入 JSON: {content}."),
            "shorten_script": Tool(name="shorten_script", func=shorten_tool, description="精简剧本。输入 JSON: {content}."),
            "generate_shot_script": Tool(name="generate_shot_script", func=shot_tool, description="把剧本转换为分镜脚本。输入 JSON: {content}."),
            "generate_similar_script": Tool(name="generate_similar_script", func=similar_tool, description="基于参考风格生成原创相似剧本。输入 JSON: {instruction, reference}."),
            "extract_meta": Tool(name="extract_meta", func=meta_tool, description="提取标题、关键词、角色和评分。输入 JSON: {content}."),
            "search_scripts": Tool(name="search_scripts", func=search_tool, description="检索相关剧本或片段。输入 JSON: {query}."),
            "general_reply": Tool(name="general_reply", func=general_tool, description="处理通用中文问答。输入 JSON: {message}."),
        }
        return tools

    def _doc_context_meta(self, selected_doc_id: str | None, recent_docs: list[dict[str, Any]], user_id: str, brief: str) -> tuple[str, str]:
        selected_content = self.rag.get_document_content(user_id, selected_doc_id) if selected_doc_id else ""
        if not selected_content:
            for item in recent_docs:
                if item.get("doc_id"):
                    selected_doc_id = item.get("doc_id")
                    selected_content = self.rag.get_document_content(user_id, selected_doc_id)
                    if selected_content:
                        break
        preview = selected_content[:100] if selected_content else ""
        meta = f"是否使用文档: {bool(selected_content)}\n文档ID: {selected_doc_id or 'None'}\n文档前100字预览: {preview or 'None'}\n查询: {brief}"
        return selected_content, meta

    def _run_react_agent(self, user_id: str, message: str, context: str, top_k: int, context_meta: str, drama_id: str | None = None) -> tuple[str, list[AgentStep]]:
        llm = self._build_react_llm()
        tools = self._build_tools(user_id, context, top_k, drama_id=drama_id)
        tool_list = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools.values())
        prompt = self._build_react_prompt()
        scratchpad = ""
        steps: list[AgentStep] = []
        for _ in range(4):
            rendered = prompt.format(tools=tool_list, context=context, context_meta=context_meta, input=message, agent_scratchpad=scratchpad)
            response = llm.invoke(rendered)
            text = getattr(response, "content", str(response))
            if "Final:" in text:
                return text.split("Final:", 1)[1].strip(), steps
            action_match = re.search(r"Action\s*:\s*(.+)", text)
            input_match = re.search(r"Action Input\s*:\s*(.+)", text)
            if not action_match:
                return text.strip(), steps
            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip() if input_match else "{}"
            tool = tools.get(tool_name)
            if tool is None:
                return text.strip(), steps
            started_at = perf_counter()
            observation = tool.invoke(tool_input)
            steps.append(AgentStep(action="act", reason=f"ReAct 调用工具 {tool_name}。", tool=tool_name, input={"input": tool_input}, output={"observation": observation if isinstance(observation, str) else str(observation)}, latency_ms=(perf_counter() - started_at) * 1000))
            scratchpad += f"{text}\nObservation: {observation}\n"
        return scratchpad.strip(), steps

    def _finalize_memory(self, user_id: str, user_text: str, assistant_text: str, project_id: str | None = None) -> None:
        self.memory_service.add_user_message(user_id, user_text, project_id=project_id)
        self.memory_service.add_ai_message(user_id, assistant_text, project_id=project_id)

    def _run_task(self, user_id: str, task_type: str, instruction: str, content: str, top_k: int, script_id: str | None, selected_doc_id: str | None = None) -> dict[str, Any]:
        brief = instruction or content
        context, search_steps, history, recent_docs, summary_text = self._build_context(user_id, brief, top_k, drama_id=script_id)
        doc_content, context_meta = self._doc_context_meta(selected_doc_id, recent_docs, user_id, brief)
        if doc_content:
            context = doc_content
        steps = list(search_steps)
        last_script = self._last_assistant_script(history)
        if last_script and task_type in {"shot", "rewrite", "continue", "expand", "shorten", "meta"} and not doc_content:
            context = last_script
        if content and not doc_content:
            context = content

        if task_type == "search":
            started_at = perf_counter()
            results = self.rag.search(user_id, script_id or brief, brief, top_k=top_k)
            final_result = {"mode": "search", "results": results}
            steps.append(self._make_step("act", "执行语义检索并返回结果。", "rag.search", {"brief": brief, "top_k": top_k}, {"result_count": len(results)}, started_at))
            payload = self._format_output(task_type, steps, final_result, "auto_plan_context", started_at)
            payload.update({"goal": brief, "task_type": task_type, "script_id": script_id, "selected_doc_id": selected_doc_id})
            return payload

        started_at = perf_counter()
        if task_type == "similar":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"instruction": instruction or content or brief, "reference": self.rag.get_retriever_context(user_id, script_id or brief, brief, top_k=top_k) or context}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "similar", "content": text}
        elif task_type == "rewrite":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"content": context, "instruction": instruction or brief}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "rewrite", "content": text}
        elif task_type == "continue":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"content": context}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "continue", "content": text}
        elif task_type == "expand":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"content": context}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "expand", "content": text}
        elif task_type == "shorten":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"content": context}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "shorten", "content": text}
        elif task_type == "shot":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"content": context}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "shot", "content": text}
        elif task_type == "meta":
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"content": context}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            try:
                meta = json.loads(text)
            except Exception:
                meta = {"raw": text}
            final_result = {"mode": "meta", **meta}
        elif task_type == "auto_generate":
            params = self._extract_generation_params(brief)
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"goal": brief, "params": params}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "auto_generate", "content": text, "params": params}
        else:
            text, agent_steps = self._run_react_agent(user_id, json.dumps({"message": instruction or content or brief}, ensure_ascii=False), context, top_k, context_meta, drama_id=script_id)
            final_result = {"mode": "general", "content": text}

        steps.extend(agent_steps)
        if final_result.get("content"):
            self._finalize_memory(user_id, instruction or content or brief, final_result.get("content", ""), project_id=script_id)
            if script_id:
                self.rag.upsert_script(user_id, script_id, final_result.get("content", ""))
        if recent_docs:
            final_result["document_context"] = f"最近上传文档：{', '.join(item.get('doc_id', '') for item in recent_docs if item.get('doc_id'))}"
        payload = self._format_output(task_type, steps, final_result, "auto_plan_context", started_at)
        payload.update({"goal": brief, "task_type": task_type, "script_id": script_id, "selected_doc_id": selected_doc_id, "history_summary": summary_text})
        return payload

    def execute_tool(self, user_id: str, intent: str, instruction: str = "", content: str = "", script_id: str | None = None, top_k: int = 3, selected_doc_id: str | None = None) -> dict[str, Any]:
        return self._run_task(user_id, intent if intent != "similar" else "similar", instruction, content, top_k, script_id, selected_doc_id=selected_doc_id)

    def plan_and_execute(self, user_id: str, goal: str, brief: str | None = None, script_id: str | None = None, top_k: int = 3, selected_doc_id: str | None = None) -> dict[str, Any]:
        brief = brief or goal
        task_type = self._classify_goal(goal)
        if selected_doc_id:
            selected_content = self.rag.get_document_content(user_id, selected_doc_id)
            if selected_content:
                brief = selected_content
        return self._run_task(user_id, task_type, goal, brief, top_k, script_id, selected_doc_id=selected_doc_id)

    def run(self, user_id: str, message: str) -> str:
        return self.plan_and_execute(user_id=user_id, goal=message, brief=message).get("content", "")

    def run_with_trace(self, user_id: str, message: str, selected_doc_id: str | None = None) -> dict[str, Any]:
        return self.plan_and_execute(user_id=user_id, goal=message, brief=message, selected_doc_id=selected_doc_id)
