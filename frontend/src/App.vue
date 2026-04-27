<template>
  <div class="chat-page">
    <main class="chat-shell">
      <section class="chat-stream" ref="chatWindowRef">
        <div v-if="messages.length === 0" class="hero-card">
          <h1>短剧 AI 助手</h1>
          <p>直接输入你的需求，我会自动理解、检索、生成并流式返回结果。</p>
        </div>

        <div v-for="msg in messages" :key="msg.id" :class="['message-row', msg.role]">
          <div class="message-avatar">{{ msg.role === 'user' ? '你' : 'AI' }}</div>
          <div class="message-bubble">
            <div v-if="msg.thinking" class="thinking-indicator">AI 正在思考并检索文档中...</div>
            <details v-if="msg.role === 'assistant' && msg.trace?.thinking_steps?.length" class="trace-card">
              <summary>查看思考流程（ReAct）</summary>
              <div class="trace-list">
                <div
                  v-for="item in msg.trace.thinking_steps"
                  :key="item.step"
                  :class="['trace-item', `trace-item--${stepType(item)}`]"
                >
                  <div class="trace-header">
                    <div class="trace-title">Step {{ item.step }} · {{ item.action }}</div>
                    <span :class="['trace-tag', `trace-tag--${stepType(item)}`]">{{ stepLabel(item) }}</span>
                  </div>
                  <div class="trace-reason">{{ item.reason }}</div>
                  <div class="trace-meta">
                    <span>tool: {{ item.tool || 'n/a' }}</span>
                    <span v-if="item.latency_ms != null">{{ formatLatency(item.latency_ms) }}</span>
                  </div>
                </div>
              </div>
            </details>
            <div class="message-content" v-html="prettyMarkdown(msg.content)"></div>
            <div class="message-time">{{ msg.time }}</div>
          </div>
        </div>
      </section>

      <section class="composer-card">
        <div class="upload-panel">
          <div class="upload-copy">
            <div class="upload-title">导入文档</div>
            <div class="upload-desc">支持 PDF、Word（.docx）和 TXT，上传后会自动解析并入库。</div>
          </div>
          <div class="upload-actions">
            <input ref="uploadInputRef" type="file" accept=".pdf,.docx,.txt" @change="handleUploadFile" />
            <button class="secondary-btn" @click="triggerUpload" :disabled="uploading">选择文件</button>
            <span class="upload-hint">{{ uploadStatus }}</span>
          </div>
        </div>
        <details v-if="uploadedDocs.length" class="doc-picker">
          <summary>参考资料（默认折叠）</summary>
          <div class="doc-picker-body">
            <label>选择要使用的文档</label>
            <select v-model="selectedDocId">
              <option value="">最近上传的文档</option>
              <option v-for="doc in uploadedDocs" :key="doc.doc_id" :value="doc.doc_id">{{ doc.doc_id }}</option>
            </select>
          </div>
        </details>
        <textarea
          v-model="inputText"
          rows="4"
          :placeholder="placeholderText"
          @keydown.enter.exact.prevent="send"
        ></textarea>
        <div class="composer-actions">
          <button class="send-btn" @click="send" :disabled="loading || !inputText.trim()">发送</button>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import axios from 'axios'

const http = axios.create({ baseURL: '/api/v1' })

const loading = ref(false)
const uploading = ref(false)
const uploadStatus = ref('未选择文件')
const uploadedDocs = ref([])
const selectedDocId = ref('')
const inputText = ref('')
const messages = ref([])
const chatWindowRef = ref(null)
const uploadInputRef = ref(null)
const pendingAssistantId = ref('')
const thinkingPhase = ref('正在读取文档')

const placeholderText = computed(() => '直接输入需求，例如“帮我自动生成一个30秒反转短剧，重生逆袭爽文，大女主，校园主题”')

const now = () => new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
const pretty = (val) => JSON.stringify(val, null, 2)
const thinkingPhases = ['正在读取文档', '正在检索相关内容', '正在组织答案']
const stepType = (item) => (item?.action === 'observe' ? 'observe' : item?.action === 'act' ? 'act' : 'reason')
const stepLabel = (item) => ({ reason: '推理', observe: '观察', act: '行动' }[stepType(item)] || '步骤')
const formatLatency = (ms) => {
  const value = Number(ms)
  if (!Number.isFinite(value)) return ''
  if (value < 1000) return `${value.toFixed(value < 10 ? 1 : 0)} ms`
  return `${(value / 1000).toFixed(2)} s`
}

const pushMessage = (role, content, trace = null) => {
  const cleanTrace = trace ? { ...trace } : null
  if (cleanTrace?.final_result) {
    delete cleanTrace.final_result.document_context
  }
  if (cleanTrace?.steps) {
    cleanTrace.steps = cleanTrace.steps.map((step) => {
      const nextStep = { ...step }
      if (nextStep.output?.documents) delete nextStep.output.documents
      if (nextStep.output?.results) delete nextStep.output.results
      if (nextStep.output?.recent_documents) delete nextStep.output.recent_documents
      return nextStep
    })
  }
  const message = { id: crypto.randomUUID(), role, content, trace: cleanTrace, time: now(), thinking: false, thinkingTimer: null }
  messages.value.push(message)
  nextTick(() => {
    if (chatWindowRef.value) chatWindowRef.value.scrollTop = chatWindowRef.value.scrollHeight
  })
  return message
}

const renderMessage = (text) => {
  const escaped = String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  return escaped
    .replace(/```([\s\S]*?)```/g, '<pre class="inline-code">$1</pre>')
    .replace(/\n/g, '<br/>')
}

const prettyMarkdown = (text) => renderMessage(text)
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/\*(.+?)\*/g, '<em>$1</em>')

const formatAssistantPayload = (payload) => {
  if (!payload) return ''
  return String(payload.content || payload?.final_result?.content || payload?.reply || '')
}

const triggerUpload = () => uploadInputRef.value?.click()

const startThinkingCycle = (message) => {
  let index = 0
  message.thinking = true
  message.content = thinkingPhases[index]
  message.thinkingTimer = setInterval(() => {
    index = (index + 1) % thinkingPhases.length
    thinkingPhase.value = thinkingPhases[index]
    message.content = thinkingPhase.value
  }, 900)
}

const stopThinkingCycle = (message) => {
  if (message?.thinkingTimer) {
    clearInterval(message.thinkingTimer)
    message.thinkingTimer = null
  }
  message.thinking = false
}

const handleUploadFile = async (event) => {
  const file = event?.target?.files?.[0]
  if (!file) return
  uploading.value = true
  uploadStatus.value = `正在上传 ${file.name}...`
  try {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('user_id', 'test-user')
    const response = await http.post('/script/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    const data = response.data?.data || {}
    uploadStatus.value = data.content || `上传完成：${file.name}`
    const docListResponse = await http.get('/script/documents', { params: { user_id: 'test-user' } })
    uploadedDocs.value = docListResponse.data?.data?.items || []
    if (!selectedDocId.value) selectedDocId.value = file.name
    pushMessage('assistant', `文件 ${file.name} 已上传并入库。`)
  } catch (error) {
    uploadStatus.value = '上传失败'
    pushMessage('assistant', `文件上传失败：${error?.response?.data ? pretty(error.response.data) : (error?.message || '请求失败')}`)
  } finally {
    uploading.value = false
    if (uploadInputRef.value) uploadInputRef.value.value = ''
  }
}

const streamText = async (text) => {
  const target = messages.value.find((item) => item.id === pendingAssistantId.value)
  if (!target) return
  let index = 0
  const timer = setInterval(() => {
    target.content = text.slice(0, index + 1)
    index += 1
    if (index >= text.length) {
      clearInterval(timer)
      target.content = text
    }
  }, 12)
}

const shouldAutoPlan = (text) => /自动|生成|短剧|反转|重生|逆袭|校园|大女主|爽文|方案|检索/.test(text)
const isDateQuery = (text) => /\b(现在)?几号\b|今天是几号|今天日期|几月几号|date|today/i.test(text)

const send = async () => {
  const text = inputText.value.trim()
  if (!text || loading.value) return
  pushMessage('user', text)
  inputText.value = ''
  loading.value = true
  const thinkingMessage = pushMessage('assistant', '正在读取文档', null)
  startThinkingCycle(thinkingMessage)

  try {
    const response = isDateQuery(text)
      ? await http.post('/chat/message', { user_id: 'test-user', message: text, selected_doc_id: selectedDocId.value || '' })
      : shouldAutoPlan(text)
        ? await http.post('/script/auto-plan', { user_id: 'test-user', goal: text, brief: text, script_id: 'script-001', top_k: 3, selected_doc_id: selectedDocId.value || '' })
        : await http.post('/chat/message', { user_id: 'test-user', message: text, selected_doc_id: selectedDocId.value || '' })

    const payload = response.data?.data
    const assistantText = formatAssistantPayload(payload) || pretty(response.data)
    stopThinkingCycle(thinkingMessage)
    thinkingMessage.content = '正在输出答案'
    thinkingMessage.thinking = false
    thinkingMessage.trace = payload || null
    pendingAssistantId.value = thinkingMessage.id
    await streamText(assistantText)
  } catch (error) {
    stopThinkingCycle(thinkingMessage)
    thinkingMessage.content = `出错了：${error?.response?.data ? pretty(error.response.data) : (error?.message || '请求失败')}`
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    await http.get('/health')
  } catch {}
  pushMessage('assistant', '你好，我是你的短剧 AI 助手。直接输入你的需求，我会自动理解并给出结果。')
})
</script>
