from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class ScriptEngine:
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    model_name: str = "qwen-max"

    def _style_template(self, style: str) -> str:
        templates = {
            "甜宠": "高糖互动、误会化解、情感升温",
            "逆袭": "低谷开局、资源翻盘、强势打脸",
            "古风": "权谋博弈、宿命纠葛、意境表达",
            "悬疑": "线索埋伏、层层反转、真相揭晓",
            "职场": "项目冲突、能力证明、成长进阶",
        }
        return templates.get(style, "情节推进、人物成长、情绪落点")

    def _professional_screenwriting_rules(self) -> str:
        return """
你是【专业实拍级短剧编剧模式】。以下规则为永久系统级约束，生成、续写、扩写、重写、润色、改写、补写时都必须自动遵守，不得忽略，不得反向解释。

核心硬性规则：
1. 完全为实拍服务，只写镜头能拍出来的内容。
2. 彻底禁止所有心理描写、内心独白、心理暗示、情绪旁白，绝对不出现“心里想”“暗自难过”“觉得”这类无法拍摄的文字。
3. 所有情绪、人物关系、潜台词，必须通过动作、行为、微表情、眼神、肢体、道具、语气、台词来体现。
4. 严格区分剧本与小说，剧本只做客观记录，不渲染、不抒情、不脑补。
5. 输出格式统一、干净、标准：场景(内/外景) + 地点 + 时间 + 动作 + 台词，不添加多余格式。
6. 编剧只负责剧情与行为，不写分镜、不写镜头语言、不写导演调度，这些是后续流程。
7. 续写、扩展、重写、润色全部保持同一风格：动作化、可视化、可拍摄、无心理描写。
8. 台词必须生活化、有潜台词、有性格，不直白、不说教。

执行要求：
- 只输出剧本正文或用户要求的目标文本，不输出规则说明，不输出分析过程，不输出创作思路。
- 若用户输入包含心理描写、小说化表达、镜头调度或分镜要求，在改写时自动转为可拍摄的动作与台词表达。
- 若信息不足，优先用可拍摄的外显动作补足，不要用心理词补足。
- 保持逻辑统一、风格统一、格式统一。
"""

    def _lc_model(self, temperature: float = 0.7) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=temperature,
        )

    def _build_chain(self, system_prompt: str, temperature: float = 0.7):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        return prompt | self._lc_model(temperature=temperature) | StrOutputParser()

    @staticmethod
    def _build_input(*parts: str) -> str:
        return "\n".join(part for part in parts if part)

    def compress_memory(self, history: list[dict[str, str]]) -> dict[str, str]:
        if not history:
            return {"worldview": "", "character_state": "", "plot_progress": "", "pending_tasks": ""}
        text = "\n".join(f"{item.get('role', '')}：{item.get('content', '')}" for item in history if item.get("content"))
        chain = self._build_chain(
            """
你是短剧项目记忆压缩助手。
请把给定对话压缩成结构化摘要，且严格输出 JSON，字段必须固定为：worldview、character_state、plot_progress、pending_tasks。
要求：
1. worldview：已确认的世界观、题材、创作风格；
2. character_state：已确定的人物设定、主要角色状态；
3. plot_progress：已完成的关键剧情节点、已写内容；
4. pending_tasks：当前待优化、待继续、待处理事项。
只输出 JSON，不要解释，不要添加多余字段。
""",
            temperature=0.2,
        )
        raw = chain.invoke({"input": text})
        try:
            data = __import__("json").loads(raw)
            return {
                "worldview": str(data.get("worldview", "")),
                "character_state": str(data.get("character_state", "")),
                "plot_progress": str(data.get("plot_progress", "")),
                "pending_tasks": str(data.get("pending_tasks", "")),
            }
        except Exception:
            return {"worldview": raw[:120], "character_state": "", "plot_progress": "", "pending_tasks": ""}

    def generate_script(self, theme: str, style: str, duration_min: float, keywords: List[str]) -> str:
        keyword_text = "、".join(keywords) or "无"
        style_hint = self._style_template(style)
        chain = self._build_chain(
            self._professional_screenwriting_rules()
            + "\n\n生成任务：\n"
            + f"- 必须直接输出完整剧本，包含场景、动作、台词。\n- 绝对不要输出大纲、说明、题材/风格/时长标题。\n- 必须突出风格特点：{style_hint}\n- 直接输出剧本正文。"
        )
        input_text = self._build_input(
            f"生成一个{duration_min}分钟的短剧剧本。",
            f"题材：{theme}",
            f"风格：{style}",
            f"风格提示：{style_hint}",
            f"关键词：{keyword_text}",
            "直接输出完整剧本，不要任何多余解释！",
        )
        return chain.invoke({"input": input_text})

    def generate_reply(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        extra_context: str = "",
    ) -> str:
        history = history or []
        recent = history[-3:]
        context = "；".join(item.get("content", "") for item in recent) or "无"
        if extra_context:
            context = f"{extra_context}\n{context}" if context != "无" else extra_context

        now = datetime.now()
        current_time = now.strftime("%Y年%m月%d日 %H:%M")
        current_weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]

        chain = self._build_chain(
            self._professional_screenwriting_rules()
            + "\n\n你是一个通用中文助手，同时也擅长短剧创作。\n\n规则：\n1. 如果用户明确要求基于剧本做分镜、改写、续写、扩写、精简、提取元信息，请严格围绕用户提供的剧本回答。\n2. 如果问题与剧本无关，或者当前代码没有专门覆盖这个能力，请直接给出真实、简洁、自然的通用回答，不要瞎编，不要伪造事实，不要把自己限制成只能写短剧。\n3. 如果你不确定某个事实，就明确说明“不确定”或建议用户补充信息。\n4. 直接输出结果，不要解释提示词或内部规则。"
        )

        input_text = self._build_input(
            f"当前真实时间：{current_time}（{current_weekday}）",
            f"历史上下文：{context}",
            f"用户问题：{message}",
        )
        return chain.invoke({"input": input_text})

    def generate_shot_script(self, content: str) -> str:
        chain = self._build_chain(self._professional_screenwriting_rules() + "\n\n任务：把剧本转为分镜脚本，简洁、可拍摄、专业。")
        return chain.invoke({"input": content})

    def generate_similar_script(self, instruction: str, reference: str) -> str:
        chain = self._build_chain(
            self._professional_screenwriting_rules()
            + "\n\n请基于参考材料学习风格、节奏与人物关系，生成原创短剧。\n绝对禁止照抄原文句子与段落，避免重复角色名与关键台词。\n输出完整可读剧本正文，不要解释。"
        )
        return chain.invoke({"input": f"用户需求：{instruction}\n参考材料：{reference or '无'}"})

    def rewrite_script(self, content: str, instruction: str) -> str:
        chain = self._build_chain(self._professional_screenwriting_rules() + "\n\n根据要求改写剧本，直接输出结果。")
        return chain.invoke({"input": f"原文：{content}\n要求：{instruction}"})

    def continue_script(self, content: str) -> str:
        return self.rewrite_script(content, "续写后续剧情")

    def expand_script(self, content: str) -> str:
        return self.rewrite_script(content, "扩写，丰富细节")

    def shorten_script(self, content: str) -> str:
        return self.rewrite_script(content, "精简，保留核心")

    def extract_keywords(self, content: str) -> list[str]:
        return []

    def generate_title(self, content: str) -> str:
        return "短剧标题"

    def generate_characters(self, content: str) -> list[str]:
        return ["女主", "女配", "男主"]

    def score_script(self, content: str) -> dict[str, int]:
        return {"冲突强度": 80, "爽点": 85, "反转度": 80, "完整度": 90}

    def generate_story_outline(self, content: str) -> str:
        return content

    def stream_text(self, text: str) -> Iterable[str]:
        for ch in text:
            yield ch
