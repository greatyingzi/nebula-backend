//! nebula-mcp (Rust): 知识星云 MCP 网关。
//! 9/13 起并入 nebula-sync-rs 核心进程(同进程双端口 9200+9300), 也可独立运行。
//! 1:1 移植 nebula/nebula_mcp_server.py, 逐字节 parity (10/10, tools/mcp_parity)。

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
};
use rmcp::{
    ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{Implementation, ServerInfo},
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    tool, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub struct McpServer {
    tool_router: ToolRouter<Self>,
    base: String,
    token: String,
    client: reqwest::Client,
}

impl McpServer {
    /// 核心进程内嵌调用: 直接注入 base(核心 REST 地址)/token, 不读 env。
    pub fn new_with(base: &str, token: &str) -> Self {
        Self {
            tool_router: Self::tool_router(),
            base: base.to_string(),
            token: token.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// 独立运行: 从 NEBULA_BASE/token 文件取配置。
    pub fn from_env() -> Self {
        let base = std::env::var("NEBULA_BASE")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:9200".into());
        let token = load_token().expect("nebula token");
        Self::new_with(&base, &token)
    }

    async fn api_get(&self, path: &str, params: Vec<(&str, String)>) -> Result<Value, String> {
        let mut url = format!("{}{}", self.base, path);
        if !params.is_empty() {
            let qs: Vec<String> = params
                .iter()
                .map(|(k, v)| format!("{}={}", urlenc(k), urlenc(v)))
                .collect();
            url.push_str(&format!("?{}", qs.join("&")));
        }
        let r = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = r.status();
        let b = r.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("HTTP {} {}", status.as_u16(), b));
        }
        serde_json::from_str(&b).map_err(|e| format!("bad json: {e}"))
    }

    async fn api_post_ops(&self, op: &Value) -> Result<Value, String> {
        let body = json!({ "ops": [op] });
        let r = self
            .client
            .post(format!("{}{}", self.base, "/ops"))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = r.status();
        let b = r.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("HTTP {} {}", status.as_u16(), b));
        }
        serde_json::from_str(&b).map_err(|e| format!("bad json: {e}"))
    }

    fn fmt_items(&self, items: &Value) -> String {
        let arr = match items.as_array() {
            Some(a) => a,
            None => return "[星云无相关记忆]".into(),
        };
        if arr.is_empty() {
            return "[星云无相关记忆]".into();
        }
        let mut lines = Vec::new();
        for (i, it) in arr.iter().enumerate() {
            let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("").trim();
            let scope = it.get("scope").and_then(|v| v.as_str()).unwrap_or("");
            let score = it.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let tags = it.get("tags").and_then(|v| v.as_str()).unwrap_or("");
            let content = if content.chars().count() > 1200 {
                format!("{}…", content.chars().take(1200).collect::<String>())
            } else {
                content.to_string()
            };
            let mut head = format!("[{}] (scope={}, score={:.3}", i + 1, scope, score);
            if !tags.is_empty() {
                head.push_str(&format!(", tags={tags}"));
            }
            head.push(')');
            lines.push(format!("{head}\n{content}"));
        }
        lines.join("\n\n")
    }

    /// 会话段列表 → 可读文本 (与 fmt_items 同风格: 定位行 + 正文截断)。
    fn fmt_session_items(&self, items: &Value, with_hint: bool) -> String {
        let arr = match items.as_array() {
            Some(a) if !a.is_empty() => a,
            _ => return "[会话库无相关片段]".into(),
        };
        let mut lines = vec![format!("会话库命中 {} 段:", arr.len())];
        for (i, it) in arr.iter().enumerate() {
            let txt = it
                .get("text")
                .and_then(|v| v.as_str())
                .or_else(|| it.get("text_preview").and_then(|v| v.as_str()))
                .unwrap_or("")
                .trim();
            let meta = format!(
                "[{}] {}/{} | {} | {}",
                i + 1,
                it.get("agent").and_then(|v| v.as_str()).unwrap_or("?"),
                it.get("device").and_then(|v| v.as_str()).unwrap_or("?"),
                it.get("ts_iso").and_then(|v| v.as_str()).unwrap_or(""),
                it.get("doc_id").and_then(|v| v.as_str()).unwrap_or("")
            );
            let body: String = txt.chars().take(1500).collect();
            lines.push(format!("\n{meta}\n{body}"));
        }
        if with_hint {
            lines.push("\n要看某个会话的完整上下文 → session_rag_session(session_id=…)".into());
        }
        lines.join("\n")
    }
}

#[derive(Debug, Deserialize, Serialize, rmcp::schemars::JsonSchema)]
pub struct SearchParams {
    /// 自然语言问题或关键词
    pub query: String,
    /// 可选, 限定分类(如 work-macvm / dsv2-model / network / hermes)
    pub scope: Option<String>,
    /// 返回条数, 默认 5, 最大 10
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, rmcp::schemars::JsonSchema)]
pub struct AddParams {
    /// 记忆正文(自含结论, 不引用外部文件路径)
    pub content: String,
    /// 分类, 默认 general
    #[serde(default)]
    pub scope: Option<String>,
    /// 可选, 逗号分隔标签
    #[serde(default)]
    pub tags: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, rmcp::schemars::JsonSchema)]
pub struct SragSearchParams {
    /// 自然语言问题或关键词
    pub query: String,
    /// 段数, 默认 6, 最大 20
    #[serde(default)]
    pub limit: Option<u32>,
    /// 可选, 限定设备: desktop-mac / work-mac / work-pc / home-server
    #[serde(default)]
    pub device: Option<String>,
    /// 可选, 限定 agent: hermes / claude / codex / pi / opencode / crush / kimi
    #[serde(default)]
    pub agent: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, rmcp::schemars::JsonSchema)]
pub struct SragSessionParams {
    /// 会话 id (从指针行或 session_rag_search 结果取)
    pub session_id: String,
    /// 段数上限, 默认 20, 最大 50
    #[serde(default)]
    pub limit: Option<u32>,
}

#[tool_router]
impl McpServer {
    #[tool(name = "nebula_search", description = "在主人的知识星云(跨设备知识库: 全部会话记忆+项目经验+踩坑教训)做语义检索。\n    当问题涉及历史项目结论、踩坑、环境配置、决策依据, 或不确定时调用。\n    Args:\n        query: 自然语言问题或关键词。\n        scope: 可选, 限定分类(如 work-macvm / dsv2-model / network / hermes)。\n        limit: 返回条数, 默认 5, 最大 10。\n    ")]
    async fn nebula_search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> String {
        let limit = p.limit.unwrap_or(5).clamp(1, 10);
        let mut params: Vec<(&str, String)> =
            vec![("q", p.query.clone()), ("limit", limit.to_string())];
        if let Some(scope) = &p.scope {
            if !scope.is_empty() {
                params.push(("scope", scope.clone()));
            }
        }
        match self.api_get("/search", params).await {
            Ok(d) => self.fmt_items(d.get("items").unwrap_or(&Value::Null)),
            Err(e) => format!("[星云检索失败: {e}]"),
        }
    }

    #[tool(name = "nebula_add", description = "往知识星云写入一条记忆(跨设备共享, 其他 agent/设备可检索到)。\n\n    触发纪律(必须遵守): 仅在用户明确要求\"记住/记下/保存这条\"时调用。\n    禁止自动沉淀对话内容、禁止把每轮结论/中间过程写入——星云是主人的知识资产,\n    写多了会污染检索结果。默认情况应只读(nebula_search), 不主动写。\n\n    Args:\n        content: 记忆正文(自含结论, 不引用外部文件路径)。\n        scope: 分类, 默认 general。\n        tags: 可选, 逗号分隔标签。\n    ")]
    async fn nebula_add(
        &self,
        Parameters(p): Parameters<AddParams>,
    ) -> String {
        let content = p.content.trim();
        if content.is_empty() {
            return "[星云写入失败: content 为空]".into();
        }
        let scope = p.scope.unwrap_or_else(|| "general".into());
        let tags = p.tags.unwrap_or_default();
        let mid = format!("{:x}", Sha256::digest(format!("{content}|{scope}")))[..16].to_string();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let machine = host_name();
        let op = json!({
            "id": mid,
            "ts": ts,
            "op": "mem.add",
            "entity": "memories",
            "payload": {
                "id": mid,
                "content": content,
                "scope": scope,
                "tags": tags,
                "created_at": ts,
                "updated_at": ts
            },
            "source": format!("{machine}-mcp")
        });
        match self.api_post_ops(&op).await {
            Ok(resp) => match resp.get("ok") {
                Some(ok) => format!("[星云写入成功 id={mid} ok={}]", py_bool_repr(ok)),
                None => format!("[星云写入返回异常: {}]", py_repr(&resp)),
            },
            Err(e) => format!("[星云写入失败: {e}]"),
        }
    }

    #[tool(name = "nebula_stats", description = "返回知识星云后端健康与统计信息(总数/分类分布)。")]
    async fn nebula_stats(&self) -> String {
        match self.api_get("/stats", vec![]).await {
            Ok(st) => {
                let s = py_dumps(&st);
                s.chars().take(2000).collect()
            }
            Err(e) => format!("[星云 stats 失败: {e}]"),
        }
    }

    #[tool(name = "session_rag_search", description = "在会话 RAG 库里检索历史会话的**原文片段**(四端 Hermes/Claude Code/Codex/pi/opencode/crush/Kimi 的历史对话都已切段入库, 与知识星云的记忆是两套)。需要看「当时原文怎么说的/那次具体怎么做的」时用它; 要的是结论/经验则用 nebula_search。\n    Args:\n        query: 自然语言或关键词。\n        limit: 返回段数, 默认 6, 最大 20。\n        device: 可选, 限定设备 (desktop-mac / work-mac / work-pc / home-server)。\n        agent: 可选, 限定 agent (hermes / claude / codex / pi / opencode / crush / kimi)。\n    ")]
    async fn session_rag_search(&self, Parameters(p): Parameters<SragSearchParams>) -> String {
        let q = p.query.trim();
        if q.is_empty() {
            return "[会话库检索失败: query 为空]".into();
        }
        let limit = p.limit.unwrap_or(6).clamp(1, 20);
        // rerank=0: hub 是 4 核老 i5, 精排 ~1.5-2s/篇 (整条查询 30s+), 而 RRF 双路已达标
        let mut params: Vec<(&str, String)> = vec![
            ("q", q.to_string()),
            ("limit", limit.to_string()),
            ("preview", "1500".into()),
            ("rerank", "0".into()),
        ];
        if let Some(d) = &p.device {
            if !d.trim().is_empty() {
                params.push(("device", d.trim().to_string()));
            }
        }
        if let Some(a) = &p.agent {
            if !a.trim().is_empty() {
                params.push(("agent", a.trim().to_string()));
            }
        }
        match self.api_get("/api/session-rag/search", params).await {
            Ok(d) => self.fmt_session_items(d.get("items").unwrap_or(&Value::Null), true),
            Err(e) => format!("[会话库检索失败: {e}]"),
        }
    }

    #[tool(name = "session_rag_session", description = "按 session_id 取出某个历史会话的**全部段落**(时间序), 用于深度下钻: 先用 session_rag_search 定位会话, 再用本工具读完整上下文。\n    Args:\n        session_id: 必填, 从会话指针行或 session_rag_search 结果里取。\n        limit: 段数上限, 默认 20, 最大 50。\n    ")]
    async fn session_rag_session(&self, Parameters(p): Parameters<SragSessionParams>) -> String {
        let sid = p.session_id.trim();
        if sid.is_empty() {
            return "[会话取回失败: session_id 为空]".into();
        }
        let limit = p.limit.unwrap_or(20).clamp(1, 50);
        let params: Vec<(&str, String)> =
            vec![("session_id", sid.to_string()), ("limit", limit.to_string())];
        match self.api_get("/api/session-rag/session", params).await {
            Ok(d) => self.fmt_session_items(d.get("items").unwrap_or(&Value::Null), false),
            Err(e) => format!("[会话取回失败: {e}]"),
        }
    }
}

impl ServerHandler for McpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        // 对齐 py mcp SDK: Implementation(name="nebula") 默认 version="" (无 instructions)
        info.server_info = Implementation::new("nebula", "")
            .with_title("知识星云 (Nebula)");
        // non-exhaustive: 经 serde from_value 构造, 键序跟随 struct 声明序 (与 py 一致)
        let caps = serde_json::from_value(json!({
            "experimental": {},
            "prompts": {"listChanged": false},
            "resources": {"subscribe": false, "listChanged": false},
            "tools": {"listChanged": false}
        }))
        .expect("server caps");
        info.capabilities = caps;
        info
    }

    /// tools/list: 手工构造 Tool, inputSchema 逐字对齐 py pydantic 生成的 JSON Schema
    /// (title/default/required/Arguments-title, 无 $schema)。工具数组顺序: search, add, stats。
    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        fn mk_tool(
            name: &str,
            desc: &str,
            schema: serde_json::Value,
        ) -> rmcp::model::Tool {
            let mut map: serde_json::Map<String, serde_json::Value> =
                serde_json::to_value(&schema)
                    .expect("schema")
                    .as_object()
                    .expect("schema obj")
                    .clone()
                    .into_iter()
                    .collect();
            let mut out = serde_json::Map::new();
            if let Some(p) = map.remove("properties") {
                out.insert("properties".into(), p);
            }
            if let Some(r) = map.remove("required") {
                out.insert("required".into(), r);
            }
            if let Some(t) = map.remove("title") {
                out.insert("title".into(), t);
            }
            if let Some(t) = map.remove("type") {
                out.insert("type".into(), t);
            }
            for (k, v) in map {
                out.insert(k, v);
            }
            let iso = serde_json::to_value(json!({
                "name": name,
                "description": desc,
                "inputSchema": out,
            }))
            .expect("tool");
            serde_json::from_value(iso).expect("tool deser")
        }

        let desc_search = "在主人的知识星云(跨设备知识库: 全部会话记忆+项目经验+踩坑教训)做语义检索。\n    当问题涉及历史项目结论、踩坑、环境配置、决策依据, 或不确定时调用。\n    Args:\n        query: 自然语言问题或关键词。\n        scope: 可选, 限定分类(如 work-macvm / dsv2-model / network / hermes)。\n        limit: 返回条数, 默认 5, 最大 10。\n    ";
        let desc_add = "往知识星云写入一条记忆(跨设备共享, 其他 agent/设备可检索到)。\n\n    触发纪律(必须遵守): 仅在用户明确要求\"记住/记下/保存这条\"时调用。\n    禁止自动沉淀对话内容、禁止把每轮结论/中间过程写入——星云是主人的知识资产,\n    写多了会污染检索结果。默认情况应只读(nebula_search), 不主动写。\n\n    Args:\n        content: 记忆正文(自含结论, 不引用外部文件路径)。\n        scope: 分类, 默认 general。\n        tags: 可选, 逗号分隔标签。\n    ";
        let desc_stats = "返回知识星云后端健康与统计信息(总数/分类分布)。";
        let desc_srag_search = "在会话 RAG 库里检索历史会话的**原文片段**(四端 Hermes/Claude Code/Codex/pi/opencode/crush/Kimi 的历史对话都已切段入库, 与知识星云的记忆是两套)。\n    需要看「当时原文怎么说的/那次具体怎么做的」时用它; 要的是结论/经验则用 nebula_search。\n    Args:\n        query: 自然语言或关键词。\n        limit: 返回段数, 默认 6, 最大 20。\n        device: 可选, 限定设备 (desktop-mac / work-mac / work-pc / home-server)。\n        agent: 可选, 限定 agent (hermes / claude / codex / pi / opencode / crush / kimi)。\n    ";
        let desc_srag_session = "按 session_id 取出某个历史会话的**全部段落**(时间序), 用于深度下钻: 先用 session_rag_search 定位会话, 再用本工具读完整上下文。\n    Args:\n        session_id: 必填, 从会话指针行或 session_rag_search 结果里取。\n        limit: 段数上限, 默认 20, 最大 50。\n    ";

        let tools = vec![
            mk_tool(
                "nebula_search",
                desc_search,
                json!({
                    "properties": {
                        "query": {"title": "Query", "type": "string"},
                        "scope": {"default": "", "title": "Scope", "type": "string"},
                        "limit": {"default": 5, "title": "Limit", "type": "integer"}
                    },
                    "required": ["query"],
                    "title": "nebula_searchArguments",
                    "type": "object"
                }),
            ),
            mk_tool(
                "nebula_add",
                desc_add,
                json!({
                    "properties": {
                        "content": {"title": "Content", "type": "string"},
                        "scope": {"default": "general", "title": "Scope", "type": "string"},
                        "tags": {"default": "", "title": "Tags", "type": "string"}
                    },
                    "required": ["content"],
                    "title": "nebula_addArguments",
                    "type": "object"
                }),
            ),
            mk_tool(
                "nebula_stats",
                desc_stats,
                json!({
                    "properties": {},
                    "title": "nebula_statsArguments",
                    "type": "object"
                }),
            ),
            mk_tool(
                "session_rag_search",
                desc_srag_search,
                json!({
                    "properties": {
                        "query": {"title": "Query", "type": "string"},
                        "limit": {"default": 6, "title": "Limit", "type": "integer"},
                        "device": {"default": "", "title": "Device", "type": "string"},
                        "agent": {"default": "", "title": "Agent", "type": "string"}
                    },
                    "required": ["query"],
                    "title": "session_rag_searchArguments",
                    "type": "object"
                }),
            ),
            mk_tool(
                "session_rag_session",
                desc_srag_session,
                json!({
                    "properties": {
                        "session_id": {"title": "Session Id", "type": "string"},
                        "limit": {"default": 20, "title": "Limit", "type": "integer"}
                    },
                    "required": ["session_id"],
                    "title": "session_rag_sessionArguments",
                    "type": "object"
                }),
            ),
        ];
        Ok(rmcp::model::ListToolsResult {
            tools,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
        let tcc =
            rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    fn get_tool(&self, name: &str) -> Option<rmcp::model::Tool> {
        self.tool_router.get(name).cloned()
    }
}

// ---- Bearer 鉴权中间件 ----

async fn bearer_auth(
    State(token): State<String>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let expect = format!("Bearer {}", token);
    let got = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if got != expect {
        // 对齐 Python: {"error": "unauthorized"} (冒号后空格)
        let body = "{\"error\": \"unauthorized\"}".to_string();
        let resp = Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        return Ok(resp);
    }
    Ok(next.run(req).await)
}

fn urlenc(s: &str) -> String {
    let mut out = String::new();
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn load_token() -> Result<String, String> {
    if let Ok(env_tok) = std::env::var("NEBULA_TOKEN") {
        let t = env_tok.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    for path in [
        format!("{home}/nebula/token.txt"),
        format!("{home}/.hermes/nebula/token"),
    ] {
        if let Ok(s) = std::fs::read_to_string(&path) {
            let t = s.trim();
            if !t.is_empty() {
                return Ok(t.to_string());
            }
        }
    }
    Err("nebula token not found (set NEBULA_TOKEN)".into())
}

fn host_name() -> String {
    let raw = std::env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::fs::read_to_string("/proc/sys/kernel/hostname").ok())
        .unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return "host".into();
    }
    raw.split('.').next().unwrap_or(raw).to_string()
}

/// 对齐 Python json.dumps 默认行为: separators=(', ', ': '), ensure_ascii=False。
/// 工具返回文本进 LLM prompt, 逐字一致是验收红线。
fn py_dumps(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_json_str(s),
        Value::Array(a) => {
            if a.is_empty() {
                return "[]".into();
            }
            let parts: Vec<String> = a.iter().map(py_dumps).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(o) => {
            if o.is_empty() {
                return "{}".into();
            }
            let parts: Vec<String> = o
                .iter()
                .map(|(k, val)| format!("{}: {}", py_json_str(k), py_dumps(val)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python json.dumps 的字符串转义(ensure_ascii=False)。
fn py_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Python f-string 里 bool 的显示: True/False (不是 JSON 的 true/false)。
fn py_bool_repr(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        other => py_repr(other),
    }
}

/// Python repr(dict/list) 风格 (仅用于异常路径)。
fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{}'", s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => {
            format!(
                "{{{}}}",
                o.iter()
                    .map(|(k, val)| format!("'{}': {}", k, py_repr(val)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// 在独立 listener 上挂 MCP streamable-HTTP + Bearer 鉴权, 跑到底。
/// 核心进程 tokio::spawn 调用: 9200(核心 REST) + 9300(MCP) 同进程双端口。
pub async fn serve_mcp(server: Arc<McpServer>, token: &str, host: &str, port: u16) {
    let factory = {
        let s = server.clone();
        move || Ok((*s).clone())
    };
    // 关 DNS rebinding Host 白名单 (TS IP / frp 域名 Host 头要能通)
    let config = StreamableHttpServerConfig::default().disable_allowed_hosts();
    let service: StreamableHttpService<McpServer, LocalSessionManager> =
        StreamableHttpService::new(factory, Arc::new(LocalSessionManager::default()), config);

    let app = Router::new()
        .nest_service("/mcp", service)
        .layer(middleware::from_fn_with_state(
            token.to_string(),
            bearer_auth,
        ));

    eprintln!(
        "nebula-mcp(rust) listening on {host}:{port} (base={})",
        server.base
    );
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .unwrap_or_else(|e| panic!("mcp bind {host}:{port}: {e}"));
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("mcp serve");
}
