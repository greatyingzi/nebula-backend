use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Config {
    pub data: PathBuf,
    pub db: PathBuf,
    pub vec_db: PathBuf,
    pub token_file: PathBuf,
    pub port: u16,
    pub mcp_port: u16,
    pub mcp_host: String,
    pub embed_url: String,
    pub embed_dim: usize,
    pub rerank_url: String,
    pub rerank_enabled: bool,
    /// 会话 RAG 的精排开关 (独立于 mem 检索)。默认关: hub 上 bge-reranker 对会话段
    /// 约 1.5-2s/篇, 20 候选 ~36s, 而 RRF 双路已达基线 (R@5 0.974) —— 收益不抵延迟。
    /// 显式传 rerank=1 仍可强制开启 (服务在 8301)。
    pub srag_rerank: bool,
    pub dedup_vec_min: f64,
    pub dedup_jac_min: f64,
    pub dedup_top_n: usize,
    pub rrf_k: i64,
    pub search_cache_max: usize,
    pub embed_cache_ttl: i64,
    pub embed_cache_max: usize,
    pub rerank_max_chars: usize,
    pub rerank_max_docs: usize,
    // ---- 阶段三: 业务后端下沉 ----
    pub llm_url: String, // 判重/合并 LLM (OpenAI 兼容 /v1/chat/completions)
    pub llm_api_key: String,
    pub llm_model: String,
    pub admin_password: String, // 看板登录 (空=开放)
    pub session_secret: String, // 登录 cookie 签名密钥 (空→复用 admin_password)
    pub camera_dir: String, // 摄像头帧本地存储目录
    pub web_root: String, // 看板 SPA 静态目录 (阶段四)
    pub rpi_turn_password: String, // 树莓派 coturn 口令 (空=不加 TURN)
    pub turn_key_id: String, // Cloudflare TURN keyId (空=只 STUN)
    pub turn_key_api_token: String,
    pub memory_dedup_vec_min: f64, // writeMemory 向量召回阈值 (0.62)
}

fn env_str(key: &str, def: &str) -> String {
    env::var(key).unwrap_or_else(|_| def.to_string())
}
fn env_bool(key: &str, def: bool) -> bool {
    match env::var(key) {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"),
        Err(_) => def,
    }
}

fn home() -> PathBuf {
    env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/home/nebula"))
}

impl Config {
    pub fn load() -> Self {
        let data = env_str("NEBULA_DATA", &home().join("nebula").to_string_lossy());
        let data = PathBuf::from(data);
        let db = data.join("nebula.db");
        let vec_db = data.join("vectors.db");
        let token_file = PathBuf::from(env_str(
            "NEBULA_TOKEN_FILE",
            &home().join("nebula").join("token.txt").to_string_lossy(),
        ));
        let camera_dir = env_str("NEBULA_CAMERA_DIR", &data.join("camera").to_string_lossy());
        let web_root = env_str("NEBULA_WEB_ROOT", &data.join("web").to_string_lossy());
        Config {
            data,
            db,
            vec_db,
            token_file,
            port: env_str("NEBULA_PORT", "9200").parse().unwrap_or(9200),
            mcp_port: env_str("NEBULA_MCP_PORT", "9300").parse().unwrap_or(9300),
            mcp_host: env_str("NEBULA_MCP_HOST", "0.0.0.0"),
            embed_url: env_str("NEBULA_EMBED_URL", "http://127.0.0.1:8300/v1/embeddings"),
            embed_dim: env_str("NEBULA_EMBED_DIM", "1024").parse().unwrap_or(1024),
            rerank_url: env_str("NEBULA_RERANK_URL", "http://127.0.0.1:8301/v1/rerank"),
            rerank_enabled: env_bool("NEBULA_RERANK_ENABLED", true),
            srag_rerank: env_bool("NEBULA_SRAG_RERANK_ENABLED", false),
            dedup_vec_min: env_str("NEBULA_DEDUP_VEC_MIN", "0.95").parse().unwrap_or(0.95),
            dedup_jac_min: env_str("NEBULA_DEDUP_JAC_MIN", "0.80").parse().unwrap_or(0.80),
            dedup_top_n: env_str("NEBULA_DEDUP_TOP_N", "3").parse().unwrap_or(3),
            rrf_k: env_str("NEBULA_RRF_K", "60").parse().unwrap_or(60),
            search_cache_max: env_str("NEBULA_CACHE_MAX", "4096").parse().unwrap_or(4096),
            embed_cache_ttl: env_str("NEBULA_EMBED_CACHE_TTL", "600").parse().unwrap_or(600),
            embed_cache_max: env_str("NEBULA_EMBED_CACHE_MAX", "256").parse().unwrap_or(256),
            rerank_max_chars: env_str("NEBULA_RERANK_MAX_CHARS", "280").parse().unwrap_or(280),
            rerank_max_docs: env_str("NEBULA_RERANK_MAX_DOCS", "20").parse().unwrap_or(20),
            // 阶段三: LLM 判重 (默认 work-pc sglang, tailnet 可达)
            llm_url: env_str("NEBULA_LLM_URL", "http://100.94.179.42:1234/v1/chat/completions"),
            llm_api_key: env_str("NEBULA_LLM_API_KEY", ""),
            llm_model: env_str("NEBULA_LLM_MODEL", "default"),
            admin_password: env_str("NEBULA_ADMIN_PASSWORD", ""),
            session_secret: env_str("NEBULA_SESSION_SECRET", ""),
            camera_dir,
            web_root,
            rpi_turn_password: env_str("NEBULA_RPI_TURN_PASSWORD", ""),
            turn_key_id: env_str("NEBULA_TURN_KEY_ID", ""),
            turn_key_api_token: env_str("NEBULA_TURN_KEY_API_TOKEN", ""),
            memory_dedup_vec_min: env_str("NEBULA_MEMORY_DEDUP_VEC_MIN", "0.62").parse().unwrap_or(0.62),
        }
    }
}

pub fn iso_now() -> String {
    use chrono::Utc;
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

pub fn load_token(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}
