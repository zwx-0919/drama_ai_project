from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api.dependencies import get_agent, get_memory_service, get_rag_service, get_script_engine, get_settings_dep
from core.audit import AuditLogger
from core.response import success
from schemas.script import AutoPlanRequest, ChatRequest, ExportRequest, MetaRequest, ScriptGenerateRequest, ScriptUpdateRequest, ShotGenerateRequest, ToolRequest
from services.llm import ScriptEngine
from services.rag import RAGService
from utils.text import read_docx_file, read_pdf_file, read_text_file

health = APIRouter()
script = APIRouter()
chat = APIRouter()
_audit = AuditLogger()


@health.get("/health")
def health_check():
    return success({"status": "ok"})


@health.get("/health/redis")
def health_redis(rag: RAGService = Depends(get_rag_service), memory=Depends(get_memory_service)):
    return success(
        {
            "memory": {
                "enabled": bool(getattr(memory, "_client", None)),
                "sample_messages": memory.get_messages("test_user")[:2],
            },
            "rag": rag.cache_status(),
        }
    )


@health.get("/health/milvus")
def health_milvus(rag: RAGService = Depends(get_rag_service)):
    collection = getattr(rag, "_collection", None)
    collection_name = getattr(rag, "collection_name", None)
    return success(
        {
            "milvus_enabled": bool(collection is not None),
            "collection_name": collection_name,
            "collection_loaded": bool(collection is not None),
            "embedding_dim": getattr(rag, "embedding_dim", None),
            "cache_status": rag.cache_status(),
        }
    )


@script.post("/generate")
def generate_script(
    req: ScriptGenerateRequest,
    engine: ScriptEngine = Depends(get_script_engine),
    rag: RAGService = Depends(get_rag_service),
):
    content = engine.generate_script(req.theme, req.style, req.duration_min, req.keywords)
    script_id = f"{req.user_id}-{req.theme}-{req.duration_min}m"
    rag.upsert_script(req.user_id, script_id, content)
    _audit.log("generate_script", {"user_id": req.user_id, "script_id": script_id})
    return success(
        {
            "thinking": [
                "reason: 解析生成参数并识别为剧本生成任务。",
                "act: 调用 llm.generate_script 生成完整剧本并写入知识库。",
            ],
            "intent": "auto_generate",
            "context_used": "request_params",
            "tool_used": ["llm.generate_script", "rag.upsert_script"],
            "content": content,
            "final_result": {"mode": "auto_generate", "script_id": script_id, "content": content},
        }
    )


@script.post("/update")
def update_script(req: ScriptUpdateRequest, rag: RAGService = Depends(get_rag_service)):
    rag.upsert_script(req.user_id, req.script_id, req.content)
    _audit.log("update_script", {"user_id": req.user_id, "script_id": req.script_id})
    return success({"script_id": req.script_id, "updated": True})


@script.post("/rewrite")
def rewrite_script(req: ToolRequest, agent=Depends(get_agent)):
    result = agent.execute_tool(user_id=req.user_id, intent="rewrite", instruction=req.instruction, content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    _audit.log("rewrite_script", {"user_id": req.user_id, "script_id": req.script_id, "instruction": req.instruction})
    return success(result)


@script.post("/continue")
def continue_script(req: ToolRequest, agent=Depends(get_agent)):
    result = agent.execute_tool(user_id=req.user_id, intent="continue", instruction=req.instruction or "续写后续剧情", content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    _audit.log("continue_script", {"user_id": req.user_id, "script_id": req.script_id})
    return success(result)


@script.post("/expand")
def expand_script(req: ToolRequest, agent=Depends(get_agent)):
    result = agent.execute_tool(user_id=req.user_id, intent="expand", instruction=req.instruction or "扩写并丰富细节", content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    _audit.log("expand_script", {"user_id": req.user_id, "script_id": req.script_id})
    return success(result)


@script.post("/shorten")
def shorten_script(req: ToolRequest, agent=Depends(get_agent)):
    result = agent.execute_tool(user_id=req.user_id, intent="shorten", instruction=req.instruction or "精简并保留核心", content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    _audit.log("shorten_script", {"user_id": req.user_id, "script_id": req.script_id})
    return success(result)


@script.post("/meta")
def script_meta(req: MetaRequest, agent=Depends(get_agent)):
    data = agent.execute_tool(user_id=req.user_id, intent="meta", instruction="提取元信息", content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    _audit.log("script_meta", {"user_id": req.user_id, "script_id": req.script_id})
    return success(data)


@script.post("/similar")
def generate_similar(req: ToolRequest, agent=Depends(get_agent)):
    data = agent.execute_tool(user_id=req.user_id, intent="similar", instruction=req.instruction or "生成相似风格短剧", content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    _audit.log("similar_script", {"user_id": req.user_id, "script_id": req.script_id})
    return success(data)


@script.post("/shot")
def generate_shot(req: ShotGenerateRequest, agent=Depends(get_agent)):
    data = agent.execute_tool(user_id=req.user_id, intent="shot", instruction="生成分镜脚本", content=req.content or "", script_id=req.script_id, selected_doc_id=getattr(req, "selected_doc_id", None))
    return success(data)


@script.get("/search")
def search_scripts(user_id: str, query: str, top_k: int = 3, agent=Depends(get_agent)):
    data = agent.execute_tool(user_id=user_id, intent="search", instruction=query, top_k=top_k)
    return success(data)


@script.get("/documents")
def list_documents(user_id: str, rag: RAGService = Depends(get_rag_service)):
    return success({"items": rag.get_recent_documents(user_id, limit=20)})


@script.post("/auto-plan")
def auto_plan(req: AutoPlanRequest, agent=Depends(get_agent)):
    result = agent.plan_and_execute(user_id=req.user_id, goal=req.goal, brief=req.brief, script_id=req.script_id, top_k=req.top_k, selected_doc_id=req.selected_doc_id)
    _audit.log("auto_plan", {"user_id": req.user_id, "goal": req.goal})
    return success(result)


@script.post("/upload")
async def upload_doc(file: UploadFile = File(...), user_id: str = Form(default="_docs"), rag: RAGService = Depends(get_rag_service), settings=Depends(get_settings_dep)):
    raw = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    temp_path = Path(settings.data_dir) / "uploads" / (file.filename or "upload.txt")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(raw)
    if suffix == ".pdf":
        content = read_pdf_file(str(temp_path))
    elif suffix == ".docx":
        content = read_docx_file(str(temp_path))
    elif suffix == ".txt":
        content = read_text_file(str(temp_path))
    else:
        return success({"error": "仅支持 pdf/docx/txt"}, msg="unsupported format", code=400)
    chunks = rag.add_document(user_id, file.filename or temp_path.stem, content, chunk_size=settings.max_chunk_size, overlap=settings.chunk_overlap)
    _audit.log("upload_doc", {"file_name": file.filename, "chunks": chunks, "user_id": user_id})
    return success({"thinking": ["reason: 识别上传文件格式并执行文本解析。", "act: 对文本进行分块后写入 Milvus，绑定 user_id。"], "intent": "upload_ingest", "context_used": "uploaded_file", "tool_used": ["utils.read_file", "rag.add_document"], "content": f"文件 {file.filename} 入库完成，共 {chunks} 个分块。", "final_result": {"file_name": file.filename, "chunks": chunks, "user_id": user_id, "document_context": rag.get_document_content(user_id, file.filename or temp_path.stem)[:500]}})


@script.post("/export")
def export_script(req: ExportRequest, rag: RAGService = Depends(get_rag_service)):
    content = rag.get_script(req.user_id, req.script_id)
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    base = export_dir / f"{req.script_id}"
    fmt = req.format.lower()
    if fmt == "txt":
        path = base.with_suffix(".txt")
        path.write_text(content, encoding="utf-8")
    elif fmt == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        path = base.with_suffix(".pdf")
        c = canvas.Canvas(str(path), pagesize=A4)
        y = 800
        for line in content.splitlines() or [""]:
            c.drawString(40, y, line[:90])
            y -= 18
            if y < 40:
                c.showPage()
                y = 800
        c.save()
    elif fmt == "docx":
        from docx import Document

        path = base.with_suffix(".docx")
        doc = Document()
        for line in content.splitlines() or [""]:
            doc.add_paragraph(line)
        doc.save(str(path))
    else:
        return success({"error": "unsupported format"}, msg="unsupported format", code=400)
    _audit.log("export_script", {"user_id": req.user_id, "script_id": req.script_id, "format": fmt})
    return FileResponse(str(path), filename=path.name)


@chat.post("/message")
def chat_message(req: ChatRequest, agent=Depends(get_agent)):
    reply = agent.plan_and_execute(user_id=req.user_id, goal=req.message, brief=req.message, selected_doc_id=req.selected_doc_id)
    _audit.log("chat_message", {"user_id": req.user_id, "selected_doc_id": req.selected_doc_id})
    return success(reply)


@chat.post("/stream")
def chat_stream(req: ChatRequest, agent=Depends(get_agent)):
    payload = agent.plan_and_execute(user_id=req.user_id, goal=req.message, brief=req.message, selected_doc_id=req.selected_doc_id)
    reply = payload.get("content", "")

    def event_stream():
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        for ch in reply:
            yield f"data: {ch}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")