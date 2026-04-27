from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from time import perf_counter
from typing import Any

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

    def _observe_search(self, user_id: str, brief: str, top_k: int) -> tuple[list[dict[str, Any]], AgentStep]:
        start = perf_counter()
        results = self.rag.search(user_id, brief, top_k=top_k)
        latency_ms = (perf_counter() - start) * 1000
        step = AgentStep(
            action="observe",
            reason="检索与目标最相关的历史内容，作为生成上下文。",
            tool="rag.search",
            input={"user_id": user_id, "query": brief, "top_k": top_k},
            output={"results": results},
            latency_ms=latency_ms,
        )
        return results, step

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

        theme_parts = [item for item in ["校园", "重生", "逆袭", "大女主", "悬疑", "爱情", "豪门", "家庭", "职场"] if
                       item in text]
        theme = "".join(theme_parts[:3]) or "短剧"
        return {
            "theme": theme,
            "style": style,
            "duration_min": duration_min,
            "keywords": keywords[:8],
        }

    @staticmethod
    def _quality_score(text: str) -> int:
        score = 50
        if len(text) >= 120:
            score += 15
        if any(token in text for token in ["冲突", "反转", "打脸", "升级"]):
            score += 15
        if any(token in text for token in ["人物", "场景", "分镜", "标题"]):
            score += 10
        return min(score, 100)

    def _normalize_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                normalized.append({"role": role, "content": content})
        return normalized

    def _last_assistant_script(self, history: list[dict[str, str]]) -> str:
        for item in reversed(history):
            if item.get("role") == "assistant":
                content = item.get("content", "")
                if content:
                    return content
        return ""

    def _build_context(self, user_id: str, brief: str, top_k: int) -> tuple[str, list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
        history = self._normalize_history(self.memory_service.get_messages(user_id))
        recent_docs = self.rag.get_recent_documents(user_id, limit=5)
        search_results, search_step = self._observe_search(user_id, brief, top_k)
        context_parts = [brief]
        if recent_docs:
            context_parts.insert(0, "\n".join(f"[上传文件] {item.get('doc_id')}: {item.get('preview', '')}" for item in recent_docs))
        if history:
            context_parts.extend([item["content"] for item in history[-6:]])
        if search_results:
            context_parts.insert(0, search_results[0].get("content", ""))
        return "\n\n".join(part for part in context_parts if part), [search_step], history, recent_docs

    def _thinking(self, step: AgentStep) -> str:
        return f"{step.action}: {step.reason}"

    def _timed_step(self, step: AgentStep, started_at: float) -> AgentStep:
        step.latency_ms = (perf_counter() - started_at) * 1000
        return step

    def _make_step(
        self,
        action: str,
        reason: str,
        tool: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        started_at: float,
    ) -> AgentStep:
        return AgentStep(
            action=action,
            reason=reason,
            tool=tool,
            input=input_data,
            output=output_data,
            latency_ms=(perf_counter() - started_at) * 1000,
        )

    def _format_output(
        self,
        intent: str,
        steps: list[AgentStep],
        final_result: dict[str, Any],
        context_used: str,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        content = final_result.get("content")
        if content is None and intent == "search":
            content = "已完成检索，请查看 results。"
        thinking_steps: list[dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            thinking_steps.append(
                {
                    "step": idx,
                    "action": step.action,
                    "reason": step.reason,
                    "tool": step.tool,
                    "input": step.input,
                    "output_summary": list(step.output.keys()) if isinstance(step.output, dict) else [],
                    "latency_ms": round(step.latency_ms, 2) if step.latency_ms is not None else None,
                }
            )
        total_latency_ms = None if started_at is None else round((perf_counter() - started_at) * 1000, 2)
        return {
            "thinking": [self._thinking(step) for step in steps],
            "thinking_steps": thinking_steps,
            "intent": intent,
            "context_used": context_used,
            "tool_used": [step.tool for step in steps if step.tool],
            "content": content or "",
            "final_result": final_result,
            "total_latency_ms": total_latency_ms,
            "steps": [step.__dict__ for step in steps],
        }

    def _resolve_tool_context(
        self,
        user_id: str,
        brief: str,
        task_type: str,
        top_k: int,
    ) -> tuple[str, list[AgentStep], str]:
        history = self._normalize_history(self.memory_service.get_messages(user_id))
        recent_docs = self.rag.get_recent_documents(user_id, limit=5)
        doc_context = "\n".join(f"[上传文件] {item.get('doc_id')}: {item.get('preview', '')}" for item in recent_docs)
        steps: list[AgentStep] = [
            AgentStep(
                action="reason",
                reason=f"解析用户需求并识别意图为 {task_type}。",
                tool="intent.classifier",
                input={"brief": brief},
                output={"intent": task_type, "recent_documents": [item.get('doc_id') for item in recent_docs]},
                latency_ms=0.0,
            )
        ]
        last_script = self._last_assistant_script(history)
        if not last_script and doc_context:
            steps.append(
                AgentStep(
                    action="observe",
                    reason="优先使用最近上传的文档作为显式上下文。",
                    tool="rag.get_recent_documents",
                    input={"user_id": user_id},
                    output={"documents": [item.get('doc_id') for item in recent_docs]},
                    latency_ms=0.0,
                )
            )
            return doc_context, steps, "recent_uploaded_docs"

        if last_script:
            steps.append(
                AgentStep(
                    action="observe",
                    reason="读取 Redis 历史中的上一轮剧本作为主要上下文。",
                    tool="memory.get_messages",
                    input={"user_id": user_id},
                    output={"has_history": True},
                    latency_ms=0.0,
                )
            )
            return last_script, steps, "redis_history"

        search_results, search_step = self._observe_search(user_id, brief, top_k)
        steps.append(
            AgentStep(
                action="observe",
                reason="Redis 无可用剧本，转为检索 Milvus 作为参考。",
                tool="fallback.rag_search",
                input={"query": brief, "top_k": top_k},
                output={"result_count": len(search_results)},
                latency_ms=0.0,
            )
        )
        steps.append(search_step)
        context = "\n\n".join(item.get("content", "") for item in search_results if item.get("content"))
        return context or brief, steps, "milvus_search" if search_results else "user_input_only"

    def execute_tool(
        self,
        user_id: str,
        intent: str,
        instruction: str = "",
        content: str = "",
        script_id: str | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        task_type = "similar" if intent == "similar" else intent
        context, steps, context_used = self._resolve_tool_context(user_id, content or instruction, task_type, top_k)
        if content:
            context = content
            context_used = "request_content"

        final_result: dict[str, Any]
        if task_type == "rewrite":
            started_at = perf_counter()
            result = self.engine.rewrite_script(context, instruction)
            final_result = {"mode": "rewrite", "content": result}
            steps.append(self._make_step("act", "调用改写工具完成重写。", "llm.rewrite_script", {"instruction": instruction}, {"content": result}, started_at))
        elif task_type == "continue":
            started_at = perf_counter()
            result = self.engine.continue_script(context)
            final_result = {"mode": "continue", "content": result}
            steps.append(self._make_step("act", "调用续写工具延展剧情。", "llm.continue_script", {}, {"content": result}, started_at))
        elif task_type == "expand":
            started_at = perf_counter()
            result = self.engine.expand_script(context)
            final_result = {"mode": "expand", "content": result}
            steps.append(self._make_step("act", "调用扩写工具丰富细节。", "llm.expand_script", {}, {"content": result}, started_at))
        elif task_type == "shorten":
            started_at = perf_counter()
            result = self.engine.shorten_script(context)
            final_result = {"mode": "shorten", "content": result}
            steps.append(self._make_step("act", "调用精简工具压缩文本。", "llm.shorten_script", {}, {"content": result}, started_at))
        elif task_type == "shot":
            started_at = perf_counter()
            result = self.engine.generate_shot_script(context)
            final_result = {"mode": "shot", "content": result}
            steps.append(self._make_step("act", "调用分镜工具转换为镜头脚本。", "llm.generate_shot_script", {}, {"content": result}, started_at))
        elif task_type == "meta":
            started_at = perf_counter()
            meta = {
                "mode": "meta",
                "title": self.engine.generate_title(context),
                "keywords": self.engine.extract_keywords(context),
                "characters": self.engine.generate_characters(context),
                "score": self.engine.score_script(context),
            }
            final_result = meta
            steps.append(self._make_step("act", "调用元信息工具提取标题/关键词/角色/评分。", "llm.meta_tools", {}, meta, started_at))
        elif task_type == "similar":
            started_at = perf_counter()
            references = self.rag.search(user_id, instruction or content or context, top_k=top_k)
            sample = "\n\n".join(item.get("content", "") for item in references if item.get("content"))
            result = self.engine.generate_similar_script(instruction or "生成相似风格短剧", sample or context)
            final_result = {"mode": "similar", "content": result, "references": references}
            steps.append(self._make_step("act", "调用相似剧本工具，基于检索参考生成原创文本。", "llm.generate_similar_script", {"reference_count": len(references)}, {"content": result}, started_at))
        elif task_type == "search":
            started_at = perf_counter()
            results = self.rag.search(user_id, instruction or content or context, top_k=top_k)
            final_result = {"mode": "search", "results": results}
            steps.append(self._make_step("act", "执行语义检索并返回结果。", "rag.search", {"top_k": top_k}, {"result_count": len(results)}, started_at))
        else:
            started_at = perf_counter()
            reply = self.engine.generate_reply(instruction or content or context, self.memory_service.get_messages(user_id), extra_context=context)
            final_result = {"mode": "general", "content": reply}
            steps.append(self._make_step("act", "未命中的专用流程，直接交给通用大模型回答。", "llm.generate_reply", {"message": instruction or content or context}, {"content": reply}, started_at))

        generated = final_result.get("content")
        if generated:
            self.memory_service.add_user_message(user_id, instruction or content)
            self.memory_service.add_ai_message(user_id, generated)
            if script_id:
                self.rag.upsert_script(user_id, script_id, generated)
        return self._format_output(task_type, steps, final_result, context_used, started_at)

    def plan_and_execute(self, user_id: str, goal: str, brief: str | None = None, script_id: str | None = None,
                         top_k: int = 3, selected_doc_id: str | None = None) -> dict[str, Any]:
        started_at = perf_counter()
        brief = brief or goal
        task_type = self._classify_goal(goal)
        steps: list[AgentStep] = []
        if task_type == "general":
            reply = self.engine.generate_reply(goal, self.memory_service.get_messages(user_id))
            final_result = {"mode": "general", "content": reply}
            steps.append(AgentStep(action="act", reason="未命中短剧工作流，直接交给通用大模型回答。", tool="llm.generate_reply", input={"goal": goal}, output={"content": reply}, latency_ms=0.0))
            self.memory_service.add_user_message(user_id, goal)
            self.memory_service.add_ai_message(user_id, reply)
            payload = self._format_output(task_type, steps, final_result, "general_llm", started_at)
            payload.update({"goal": goal, "task_type": task_type, "script_id": script_id})
            return payload
        context, search_steps, history, recent_docs = self._build_context(user_id, brief, top_k)
        steps.extend(search_steps)
        last_script = self._last_assistant_script(history)
        if selected_doc_id:
            selected_content = self.rag.get_document_content(user_id, selected_doc_id)
            if selected_content:
                context = selected_content
                document_context = f"指定文档：{selected_doc_id}"
            else:
                document_context = f"指定文档：{selected_doc_id}（未找到内容，回退到默认上下文）"
        else:
            uploaded_names = ", ".join(item.get("doc_id", "") for item in recent_docs if item.get("doc_id"))
            document_context = f"最近上传文档：{uploaded_names}" if uploaded_names else ""
        if last_script and task_type in {"shot", "rewrite", "continue", "expand", "shorten", "meta"} and not selected_doc_id:
            context = last_script

        final_result: dict[str, Any]
        if task_type == "search":
            started_at = perf_counter()
            final_result = {"mode": "search", "results": self.rag.search(user_id, brief, top_k=top_k)}
            steps.append(self._make_step("act", "执行语义检索并返回结果。", "rag.search", {"brief": brief, "top_k": top_k}, {"result_count": len(final_result["results"])}, started_at))
        elif task_type == "similar":
            started_at = perf_counter()
            references = self.rag.search(user_id, brief, top_k=top_k)
            ref_text = "\n\n".join(item.get("content", "") for item in references if item.get("content"))
            result = self.engine.generate_similar_script(goal, ref_text or context)
            final_result = {"mode": "similar", "content": result, "references": references}
            steps.append(self._make_step("act", "基于检索参考生成相似风格原创剧本。", "llm.generate_similar_script", {"goal": goal}, {"content": result}, started_at))
        elif task_type == "rewrite":
            started_at = perf_counter()
            result = self.engine.rewrite_script(context, goal)
            final_result = {"mode": "rewrite", "content": result}
            steps.append(self._make_step("act", "基于上一轮剧本进行改写。", "llm.rewrite_script", {"content": context, "instruction": goal}, {"content": result}, started_at))
        elif task_type == "continue":
            started_at = perf_counter()
            result = self.engine.continue_script(context)
            final_result = {"mode": "continue", "content": result}
            steps.append(self._make_step("act", "基于上一轮剧本继续生成。", "llm.continue_script", {"content": context}, {"content": result}, started_at))
        elif task_type == "expand":
            started_at = perf_counter()
            result = self.engine.expand_script(context)
            final_result = {"mode": "expand", "content": result}
            steps.append(self._make_step("act", "基于上一轮剧本扩写。", "llm.expand_script", {"content": context}, {"content": result}, started_at))
        elif task_type == "shorten":
            started_at = perf_counter()
            result = self.engine.shorten_script(context)
            final_result = {"mode": "shorten", "content": result}
            steps.append(self._make_step("act", "基于上一轮剧本精简。", "llm.shorten_script", {"content": context}, {"content": result}, started_at))
        elif task_type == "shot":
            started_at = perf_counter()
            result = self.engine.generate_shot_script(context)
            final_result = {"mode": "shot", "content": result}
            steps.append(self._make_step("act", "必须基于上一轮剧本生成分镜。", "llm.generate_shot_script", {"content": context}, {"content": result}, started_at))
        elif task_type == "meta":
            started_at = perf_counter()
            meta = {"mode": "meta", "title": self.engine.generate_title(context), "keywords": self.engine.extract_keywords(context), "characters": self.engine.generate_characters(context), "score": self.engine.score_script(context)}
            final_result = meta
            steps.append(self._make_step("act", "提取元信息。", "llm.meta_tools", {"content": context}, meta, started_at))
        else:
            started_at = perf_counter()
            params = self._extract_generation_params(goal)
            script = self.engine.generate_script(params["theme"], params["style"], params["duration_min"], params["keywords"])
            final_result = {"mode": "auto_generate", "content": script, "params": params}
            steps.append(self._make_step("act", "根据目标直接生成完整短剧剧本。", "llm.generate_script", params, {"content": script}, started_at))

        if final_result.get("mode") in {"rewrite", "continue", "expand", "shorten", "shot", "auto_generate", "general"}:
            self.memory_service.add_user_message(user_id, goal)
            self.memory_service.add_ai_message(user_id, final_result.get("content", ""))

        if document_context:
            final_result["document_context"] = document_context
        payload = self._format_output(task_type, steps, final_result, "auto_plan_context", started_at)
        payload.update({"goal": goal, "task_type": task_type, "script_id": script_id})
        return payload

    def run(self, user_id: str, message: str) -> str:
        history = self.memory_service.get_messages(user_id)
        reply = self.engine.generate_reply(message, history)
        self.memory_service.add_user_message(user_id, message)
        self.memory_service.add_ai_message(user_id, reply)
        return reply

    def run_with_trace(self, user_id: str, message: str) -> dict[str, Any]:
        started_at = perf_counter()
        history = self.memory_service.get_messages(user_id)
        recent_docs = self.rag.get_recent_documents(user_id, limit=5)
        doc_context = "\n".join(f"[上传文件] {item.get('doc_id')}: {item.get('preview', '')}" for item in recent_docs)
        reply = self.engine.generate_reply(message, history, extra_context=doc_context)
        self.memory_service.add_user_message(user_id, message)
        self.memory_service.add_ai_message(user_id, reply)
        steps = [
            AgentStep(action="reason", reason="解析聊天问题并确认回复目标。", tool="intent.chat", input={"message": message}, output={"mode": "general_or_chat"}, latency_ms=0.0),
            AgentStep(action="observe", reason="读取 Redis 对话历史与最近上传文档作为上下文。", tool="memory.get_messages", input={"user_id": user_id}, output={"history_count": len(history), "document_count": len(recent_docs)}, latency_ms=0.0),
            AgentStep(action="act", reason="调用通用对话工具生成最终回复。", tool="llm.generate_reply", input={"message": message}, output={"content": reply}, latency_ms=0.0),
        ]
        final_result = {"mode": "chat", "content": reply}
        if doc_context:
            final_result["document_context"] = doc_context
        return self._format_output("chat", steps, final_result, "recent_uploaded_docs" if recent_docs else "redis_history", started_at)
