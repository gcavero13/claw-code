//! Live API trace test — sends a real user message to the Anthropic API,
//! captures every `ApiRequest` at each iteration, and writes them to a
//! JSON file for later inspection.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo test -p runtime --test live_api_trace_test -- --nocapture --ignored
//!
//! The trace is written to: rust/target/api_trace_output.json

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use runtime::{
    ApiClient, ApiRequest, AssistantEvent, ContentBlock, ConversationMessage, ConversationRuntime,
    MessageRole, PermissionMode, PermissionPolicy, RuntimeError, Session, StaticToolExecutor,
    SystemPromptBuilder, ProjectContext, TokenUsage,
};
use serde_json::{json, Value};

// ─── Snapshot types ───

#[derive(Debug, Clone)]
struct IterationSnapshot {
    iteration: usize,
    system_prompt_segments: usize,
    system_prompt_total_chars: usize,
    messages: Vec<SnapshotMessage>,
}

#[derive(Debug, Clone)]
struct SnapshotMessage {
    role: String,
    blocks: Vec<Value>,
}

fn snapshot_message(msg: &ConversationMessage) -> SnapshotMessage {
    let role = match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };
    let blocks = msg
        .blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => json!({
                "type": "text",
                "text": text
            }),
            ContentBlock::ToolUse { id, name, input } => json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            }),
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
            } => json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "output": output,
                "is_error": is_error
            }),
        })
        .collect();
    SnapshotMessage {
        role: role.to_string(),
        blocks,
    }
}

fn snapshot_to_json(snap: &IterationSnapshot) -> Value {
    json!({
        "iteration": snap.iteration,
        "system_prompt_segments": snap.system_prompt_segments,
        "system_prompt_total_chars": snap.system_prompt_total_chars,
        "message_count": snap.messages.len(),
        "messages": snap.messages.iter().map(|m| json!({
            "role": &m.role,
            "blocks": &m.blocks,
        })).collect::<Vec<_>>()
    })
}

// ─── Live API client that spies on every request ───

struct LiveSpyApiClient {
    rt: tokio::runtime::Runtime,
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    tool_specs: Vec<Value>,
    call_count: usize,
    snapshots: Arc<Mutex<Vec<IterationSnapshot>>>,
}

impl LiveSpyApiClient {
    fn new(
        api_key: String,
        model: String,
        tool_specs: Vec<Value>,
        snapshots: Arc<Mutex<Vec<IterationSnapshot>>>,
    ) -> Self {
        let base_url = env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        Self {
            rt: tokio::runtime::Runtime::new().expect("tokio runtime"),
            http: reqwest::Client::new(),
            api_key,
            base_url,
            model,
            tool_specs,
            call_count: 0,
            snapshots,
        }
    }

    /// Convert runtime messages to the Anthropic API format.
    fn convert_messages(messages: &[ConversationMessage]) -> Vec<Value> {
        messages
            .iter()
            .filter_map(|message| {
                let role = match message.role {
                    MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
                    MessageRole::Assistant => "assistant",
                };
                let content: Vec<Value> = message
                    .blocks
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text { text } => json!({
                            "type": "text",
                            "text": text
                        }),
                        ContentBlock::ToolUse { id, name, input } => {
                            let input_value: Value = serde_json::from_str(input)
                                .unwrap_or_else(|_| json!({ "raw": input }));
                            json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input_value
                            })
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                            ..
                        } => json!({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": [{ "type": "text", "text": output }],
                            "is_error": is_error
                        }),
                    })
                    .collect();
                if content.is_empty() {
                    None
                } else {
                    Some(json!({ "role": role, "content": content }))
                }
            })
            .collect()
    }
}

impl ApiClient for LiveSpyApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        self.call_count += 1;
        let iteration = self.call_count;

        // ── Capture the snapshot ──
        let snapshot = IterationSnapshot {
            iteration,
            system_prompt_segments: request.system_prompt.len(),
            system_prompt_total_chars: request.system_prompt.iter().map(|s| s.len()).sum(),
            messages: request.messages.iter().map(snapshot_message).collect(),
        };
        self.snapshots.lock().unwrap().push(snapshot);

        // ── Build the real API request ──
        let system_text = request.system_prompt.join("\n\n");
        let api_messages = Self::convert_messages(&request.messages);
        let mut body = json!({
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        });
        if !system_text.is_empty() {
            body["system"] = json!(system_text);
        }
        if !self.tool_specs.is_empty() {
            body["tools"] = json!(self.tool_specs);
            body["tool_choice"] = json!({"type": "auto"});
        }

        // ── Make the real HTTP call (non-streaming for simplicity) ──
        let response_body: Value = self.rt.block_on(async {
            let resp = self
                .http
                .post(format!("{}/v1/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| RuntimeError::new(format!("HTTP error: {e}")))?;

            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|e| RuntimeError::new(format!("read body: {e}")))?;

            if !status.is_success() {
                return Err(RuntimeError::new(format!(
                    "API returned {status}: {text}"
                )));
            }

            serde_json::from_str::<Value>(&text)
                .map_err(|e| RuntimeError::new(format!("parse JSON: {e}")))
        })?;

        // ── Convert response to AssistantEvents ──
        let mut events = Vec::new();
        if let Some(content) = response_body["content"].as_array() {
            for block in content {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            events.push(AssistantEvent::TextDelta(text.to_string()));
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("unknown").to_string();
                        let name = block["name"].as_str().unwrap_or("unknown").to_string();
                        let input = block["input"].to_string();
                        events.push(AssistantEvent::ToolUse { id, name, input });
                    }
                    _ => {}
                }
            }
        }

        // Extract usage
        if let Some(usage) = response_body.get("usage") {
            events.push(AssistantEvent::Usage(TokenUsage {
                input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
                cache_creation_input_tokens: usage["cache_creation_input_tokens"]
                    .as_u64()
                    .unwrap_or(0) as u32,
                cache_read_input_tokens: usage["cache_read_input_tokens"]
                    .as_u64()
                    .unwrap_or(0) as u32,
            }));
        }

        events.push(AssistantEvent::MessageStop);
        Ok(events)
    }
}

// ─── The test ───

#[test]
#[ignore] // Only runs when explicitly requested (needs ANTHROPIC_API_KEY)
fn live_trace_api_calls_and_save_to_file() {
    let api_key = env::var("ANTHROPIC_API_KEY").expect(
        "Set ANTHROPIC_API_KEY to run this test:\n  \
         ANTHROPIC_API_KEY=sk-... cargo test -p runtime --test live_api_trace_test -- --nocapture --ignored",
    );
    let model = env::var("CLAW_TEST_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());

    // ── Define tools the model can call ──
    let tool_specs = vec![
        json!({
            "name": "grep_search",
            "description": "Search for a pattern in files. Returns matching lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regex pattern to search for" },
                    "path": { "type": "string", "description": "Directory to search in" }
                },
                "required": ["pattern"]
            }
        }),
        json!({
            "name": "read_file",
            "description": "Read a file and return its contents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the file" },
                    "offset": { "type": "integer", "description": "Line to start reading from" },
                    "limit": { "type": "integer", "description": "Max lines to read" }
                },
                "required": ["file_path"]
            }
        }),
        json!({
            "name": "list_files",
            "description": "List files matching a glob pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern like **/*.rs" }
                },
                "required": ["pattern"]
            }
        }),
    ];

    // ── Register mock tool handlers (return realistic but fake data) ──
    let tool_executor = StaticToolExecutor::new()
        .register("grep_search", |input| {
            Ok(format!(
                "src/main.rs:15: // TODO: refactor error handling\n\
                 src/auth.rs:42: // TODO: validate token expiry\n\
                 src/db.rs:88: // TODO: add connection pooling\n\
                 [searched with input: {input}]"
            ))
        })
        .register("read_file", |input| {
            Ok(format!(
                "1\tfn main() {{\n\
                 2\t    println!(\"hello\");\n\
                 3\t    // TODO: refactor error handling\n\
                 4\t    run_server();\n\
                 5\t}}\n\
                 [read with input: {input}]"
            ))
        })
        .register("list_files", |input| {
            Ok(format!(
                "src/main.rs\nsrc/auth.rs\nsrc/db.rs\nsrc/lib.rs\n[listed with input: {input}]"
            ))
        });

    let snapshots: Arc<Mutex<Vec<IterationSnapshot>>> = Arc::new(Mutex::new(Vec::new()));

    let api_client = LiveSpyApiClient::new(
        api_key,
        model.clone(),
        tool_specs,
        snapshots.clone(),
    );

    let system_prompt = SystemPromptBuilder::new()
        .with_project_context(ProjectContext {
            cwd: PathBuf::from("/tmp/example-project"),
            current_date: "2026-04-06".to_string(),
            git_status: None,
            git_diff: None,
            instruction_files: Vec::new(),
        })
        .with_os("darwin", "25.3.0")
        .build();

    let permission_policy = PermissionPolicy::new(PermissionMode::DangerFullAccess);

    let mut runtime = ConversationRuntime::new(
        Session::new(),
        api_client,
        tool_executor,
        permission_policy,
        system_prompt,
    )
    .with_max_iterations(6); // Safety cap

    // ── The user message — pick yours or override with CLAW_TEST_PROMPT ──
    let user_message = env::var("CLAW_TEST_PROMPT").unwrap_or_else(|_| {
        "Find all TODO comments in the project and tell me which one is most important to fix first."
            .to_string()
    });

    println!("\n>>> User message: {user_message}");
    println!(">>> Model: {model}");
    println!(">>> Max iterations: 6\n");

    let summary = runtime
        .run_turn(&user_message, None)
        .expect("turn should succeed");

    // ── Print to console ──
    let snaps = snapshots.lock().unwrap();
    for snap in snaps.iter() {
        println!("\n{}", "=".repeat(70));
        println!("  ITERATION {} — {} messages sent to API", snap.iteration, snap.messages.len());
        println!("{}", "=".repeat(70));
        println!(
            "  system_prompt: {} segment(s), {} chars",
            snap.system_prompt_segments, snap.system_prompt_total_chars
        );
        for (i, msg) in snap.messages.iter().enumerate() {
            println!("  [{i}] role={}", msg.role);
            for (j, block) in msg.blocks.iter().enumerate() {
                let compact = serde_json::to_string(block).unwrap_or_default();
                let display = if compact.len() > 150 {
                    format!("{}...", &compact[..150])
                } else {
                    compact
                };
                println!("       block[{j}]: {display}");
            }
        }
    }

    // ── Print final session ──
    let session = runtime.session();
    println!("\n{}", "=".repeat(70));
    println!(
        "  FINAL SESSION — {} messages, {} iterations, {} input tokens, {} output tokens",
        session.messages.len(),
        summary.iterations,
        summary.usage.input_tokens,
        summary.usage.output_tokens,
    );
    println!("{}", "=".repeat(70));
    for (i, msg) in session.messages.iter().enumerate() {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let block_summary: Vec<String> = msg
            .blocks
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => {
                    let t = if text.len() > 80 {
                        format!("{}...", &text[..80])
                    } else {
                        text.clone()
                    };
                    format!("Text({t})")
                }
                ContentBlock::ToolUse { name, .. } => format!("ToolUse({name})"),
                ContentBlock::ToolResult {
                    tool_name,
                    is_error,
                    ..
                } => {
                    let err = if *is_error { " ERROR" } else { "" };
                    format!("ToolResult({tool_name}{err})")
                }
            })
            .collect();
        println!("  [{i}] {role:<10} {}", block_summary.join(" | "));
    }

    // ── Write full trace to JSON file ──
    let trace = json!({
        "user_message": user_message,
        "model": model,
        "iterations": summary.iterations,
        "total_input_tokens": summary.usage.input_tokens,
        "total_output_tokens": summary.usage.output_tokens,
        "api_calls": snaps.iter().map(snapshot_to_json).collect::<Vec<_>>(),
        "final_session": session.messages.iter().map(|msg| {
            let role = match msg.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };
            json!({
                "role": role,
                "blocks": msg.blocks.iter().map(|b| match b {
                    ContentBlock::Text { text } => json!({"type": "text", "text": text}),
                    ContentBlock::ToolUse { id, name, input } => json!({"type": "tool_use", "id": id, "name": name, "input": input}),
                    ContentBlock::ToolResult { tool_use_id, tool_name, output, is_error } => json!({"type": "tool_result", "tool_use_id": tool_use_id, "tool_name": tool_name, "output": output, "is_error": is_error}),
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    });

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/api_trace_output.json");
    fs::write(&output_path, serde_json::to_string_pretty(&trace).unwrap())
        .expect("failed to write trace file");
    println!("\n>>> Trace written to: {}", output_path.display());
}
