//! Test that traces the exact messages sent to the API at each iteration
//! within a single turn. This makes visible what the model sees at every
//! point in the agentic loop.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use runtime::{
    ApiClient, ApiRequest, AssistantEvent, ConversationRuntime, ContentBlock, MessageRole,
    PermissionMode, PermissionPolicy, RuntimeError, Session, StaticToolExecutor, TokenUsage,
};
use runtime::{ProjectContext, SystemPromptBuilder};

/// Captured snapshot of one API call — the iteration number and the full
/// list of messages that were sent.
#[derive(Debug, Clone)]
struct IterationSnapshot {
    iteration: usize,
    system_prompt: Vec<String>,
    messages: Vec<SnapshotMessage>,
}

/// Simplified view of a message for easy assertion and printing.
#[derive(Debug, Clone)]
struct SnapshotMessage {
    role: &'static str,
    blocks: Vec<String>,
}

fn snapshot_message(msg: &runtime::ConversationMessage) -> SnapshotMessage {
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
            ContentBlock::Text { text } => format!("Text({text})"),
            ContentBlock::ToolUse { id, name, input } => {
                format!("ToolUse(id={id}, name={name}, input={input})")
            }
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
            } => {
                let err = if *is_error { ", ERROR" } else { "" };
                format!("ToolResult(id={tool_use_id}, name={tool_name}, output={output}{err})")
            }
        })
        .collect();
    SnapshotMessage { role, blocks }
}

/// An API client that records every request it receives, then returns
/// scripted responses to drive a multi-iteration turn.
struct SpyApiClient {
    call_count: usize,
    snapshots: Arc<Mutex<Vec<IterationSnapshot>>>,
}

impl SpyApiClient {
    fn new(snapshots: Arc<Mutex<Vec<IterationSnapshot>>>) -> Self {
        Self {
            call_count: 0,
            snapshots,
        }
    }
}

impl ApiClient for SpyApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        self.call_count += 1;

        // Capture the full request as a snapshot
        let snapshot = IterationSnapshot {
            iteration: self.call_count,
            system_prompt: request.system_prompt.clone(),
            messages: request.messages.iter().map(snapshot_message).collect(),
        };
        self.snapshots.lock().unwrap().push(snapshot);

        // Scripted responses for a 3-iteration turn:
        //   1. Model calls grep_search
        //   2. Model calls read_file + edit_file (two tools at once)
        //   3. Model responds with text only (turn ends)
        match self.call_count {
            1 => Ok(vec![
                AssistantEvent::TextDelta("Let me find the TODOs.".to_string()),
                AssistantEvent::ToolUse {
                    id: "tu_1".to_string(),
                    name: "grep_search".to_string(),
                    input: r#"{"pattern":"TODO","path":"src/"}"#.to_string(),
                },
                AssistantEvent::Usage(TokenUsage {
                    input_tokens: 500,
                    output_tokens: 30,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                }),
                AssistantEvent::MessageStop,
            ]),
            2 => Ok(vec![
                AssistantEvent::TextDelta("Found it. Reading and fixing.".to_string()),
                AssistantEvent::ToolUse {
                    id: "tu_2".to_string(),
                    name: "read_file".to_string(),
                    input: r#"{"path":"src/auth.rs"}"#.to_string(),
                },
                AssistantEvent::ToolUse {
                    id: "tu_3".to_string(),
                    name: "edit_file".to_string(),
                    input: r#"{"path":"src/auth.rs","old":"// TODO: validate","new":"validate_token(t)"}"#.to_string(),
                },
                AssistantEvent::Usage(TokenUsage {
                    input_tokens: 800,
                    output_tokens: 45,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                }),
                AssistantEvent::MessageStop,
            ]),
            3 => Ok(vec![
                AssistantEvent::TextDelta(
                    "Done. Replaced the TODO in src/auth.rs with a validation call.".to_string(),
                ),
                AssistantEvent::Usage(TokenUsage {
                    input_tokens: 1200,
                    output_tokens: 20,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                }),
                AssistantEvent::MessageStop,
            ]),
            _ => unreachable!("unexpected iteration {}", self.call_count),
        }
    }
}

fn print_snapshot(snap: &IterationSnapshot) {
    println!("\n============================================================");
    println!("  ITERATION {}", snap.iteration);
    println!("============================================================");
    println!(
        "  system_prompt: {} segment(s), {} chars total",
        snap.system_prompt.len(),
        snap.system_prompt.iter().map(|s| s.len()).sum::<usize>()
    );
    println!("  messages ({} total):", snap.messages.len());
    for (i, msg) in snap.messages.iter().enumerate() {
        println!("    [{i}] role={}", msg.role);
        for (j, block) in msg.blocks.iter().enumerate() {
            // Truncate long blocks for readability
            let display = if block.len() > 120 {
                format!("{}...", &block[..120])
            } else {
                block.clone()
            };
            println!("         block[{j}]: {display}");
        }
    }
}

#[test]
fn trace_messages_at_each_iteration_within_single_turn() {
    let snapshots: Arc<Mutex<Vec<IterationSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
    let api_client = SpyApiClient::new(snapshots.clone());

    // Register mock tools that return deterministic output
    let tool_executor = StaticToolExecutor::new()
        .register("grep_search", |_input| {
            Ok("src/auth.rs:10: // TODO: validate\nsrc/lib.rs:42: // TODO: cleanup".to_string())
        })
        .register("read_file", |_input| {
            Ok("line 9: fn check_token(t: &str) {\nline 10: // TODO: validate\nline 11: }".to_string())
        })
        .register("edit_file", |_input| Ok("file edited successfully".to_string()));

    let permission_policy = PermissionPolicy::new(PermissionMode::DangerFullAccess);
    let system_prompt = SystemPromptBuilder::new()
        .with_project_context(ProjectContext {
            cwd: PathBuf::from("/tmp/test-project"),
            current_date: "2026-04-06".to_string(),
            git_status: None,
            git_diff: None,
            instruction_files: Vec::new(),
        })
        .with_os("darwin", "25.3.0")
        .build();

    let mut runtime = ConversationRuntime::new(
        Session::new(),
        api_client,
        tool_executor,
        permission_policy,
        system_prompt,
    );

    let summary = runtime
        .run_turn(
            "Find all TODO comments in src/ and fix the first one",
            None,
        )
        .expect("turn should succeed");

    // ── Print every snapshot ──
    let snaps = snapshots.lock().unwrap();
    for snap in snaps.iter() {
        print_snapshot(snap);
    }

    // ── Structural assertions ──

    assert_eq!(summary.iterations, 3, "should take exactly 3 iterations");
    assert_eq!(snaps.len(), 3, "should have captured 3 API calls");

    // ── Iteration 1: only the user message ──
    let iter1 = &snaps[0];
    assert_eq!(iter1.messages.len(), 1, "iter1: just the user message");
    assert_eq!(iter1.messages[0].role, "user");
    assert_eq!(iter1.messages[0].blocks.len(), 1);
    assert!(iter1.messages[0].blocks[0].contains("TODO comments"));

    // ── Iteration 2: user + assistant(text+tool) + tool_result ──
    let iter2 = &snaps[1];
    assert_eq!(
        iter2.messages.len(),
        3,
        "iter2: user + assistant + tool_result"
    );
    // [0] user
    assert_eq!(iter2.messages[0].role, "user");
    // [1] assistant with text + tool_use in same message
    assert_eq!(iter2.messages[1].role, "assistant");
    assert_eq!(
        iter2.messages[1].blocks.len(),
        2,
        "assistant has text + tool_use"
    );
    assert!(iter2.messages[1].blocks[0].starts_with("Text("));
    assert!(iter2.messages[1].blocks[1].starts_with("ToolUse("));
    assert!(iter2.messages[1].blocks[1].contains("grep_search"));
    // [2] tool result (separate message, role=tool)
    assert_eq!(iter2.messages[2].role, "tool");
    assert_eq!(iter2.messages[2].blocks.len(), 1);
    assert!(iter2.messages[2].blocks[0].starts_with("ToolResult("));
    assert!(iter2.messages[2].blocks[0].contains("grep_search"));
    assert!(iter2.messages[2].blocks[0].contains("src/auth.rs:10"));

    // ── Iteration 3: user + asst₁ + tool₁ + asst₂(text+2tools) + tool₂a + tool₂b ──
    let iter3 = &snaps[2];
    assert_eq!(
        iter3.messages.len(),
        6,
        "iter3: user + asst + tr + asst + tr + tr"
    );
    // [0] user (same as always)
    assert_eq!(iter3.messages[0].role, "user");
    // [1] assistant from iteration 1
    assert_eq!(iter3.messages[1].role, "assistant");
    // [2] tool result from iteration 1
    assert_eq!(iter3.messages[2].role, "tool");
    // [3] assistant from iteration 2 — text + TWO tool uses
    assert_eq!(iter3.messages[3].role, "assistant");
    assert_eq!(
        iter3.messages[3].blocks.len(),
        3,
        "second assistant has text + 2 tool_uses"
    );
    assert!(iter3.messages[3].blocks[0].starts_with("Text("));
    assert!(iter3.messages[3].blocks[1].contains("read_file"));
    assert!(iter3.messages[3].blocks[2].contains("edit_file"));
    // [4] tool result for read_file
    assert_eq!(iter3.messages[4].role, "tool");
    assert!(iter3.messages[4].blocks[0].contains("read_file"));
    // [5] tool result for edit_file
    assert_eq!(iter3.messages[5].role, "tool");
    assert!(iter3.messages[5].blocks[0].contains("edit_file"));
    assert!(iter3.messages[5].blocks[0].contains("file edited successfully"));

    // ── Verify the final session state matches iteration 3 + the final assistant ──
    let session = runtime.session();
    assert_eq!(
        session.messages.len(),
        7,
        "final session: user + asst + tr + asst + tr + tr + asst(final)"
    );
    assert_eq!(session.messages[0].role, MessageRole::User);
    assert_eq!(session.messages[1].role, MessageRole::Assistant);
    assert_eq!(session.messages[2].role, MessageRole::Tool);
    assert_eq!(session.messages[3].role, MessageRole::Assistant);
    assert_eq!(session.messages[4].role, MessageRole::Tool);
    assert_eq!(session.messages[5].role, MessageRole::Tool);
    assert_eq!(session.messages[6].role, MessageRole::Assistant);

    // Final assistant message is text-only (no tool calls)
    assert!(matches!(
        &session.messages[6].blocks[0],
        ContentBlock::Text { text } if text.contains("Done")
    ));

    println!("\n\n============================================================");
    println!("  FINAL SESSION ({} messages)", session.messages.len());
    println!("============================================================");
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
                    let t = if text.len() > 60 {
                        format!("{}...", &text[..60])
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
}
