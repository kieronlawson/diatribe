use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::llm::speaker_id_prompt::get_speaker_id_tool_schema;
use crate::models::{SpeakerIdentification, WindowPatch};

/// Configuration for the Anthropic API client
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key (from ANTHROPIC_API_KEY env var)
    pub api_key: String,
    /// Model to use (e.g., "claude-haiku-4-5-20250514")
    pub model: String,
    /// Temperature (0-1, lower = more deterministic)
    pub temperature: f64,
    /// Maximum tokens in response
    pub max_tokens: u32,
}

impl AnthropicConfig {
    /// Create config from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY environment variable not set")?;

        Ok(Self {
            api_key,
            model: "claude-haiku-4-5-20251001".to_string(),
            temperature: 0.1,
            max_tokens: 4096,
        })
    }

    /// Create with custom settings
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            temperature: 0.1,
            max_tokens: 4096,
        }
    }
}

/// Log entry for API request/response logging
#[derive(Debug, Serialize)]
struct LogEntry {
    timestamp: String,
    method: String,
    duration_ms: u64,
    request: serde_json::Value,
    response: Option<serde_json::Value>,
    status_code: Option<u16>,
    error: Option<String>,
}

/// Anthropic API client
pub struct AnthropicClient {
    client: Client,
    config: AnthropicConfig,
    log_dir: Option<PathBuf>,
    log_sequence: AtomicUsize,
}

impl AnthropicClient {
    pub fn new(config: AnthropicConfig, log_dir: Option<PathBuf>) -> Self {
        // Create log directory if specified and doesn't exist
        if let Some(ref dir) = log_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                warn!("Failed to create log directory {:?}: {}", dir, e);
            }
        }

        Self {
            client: Client::new(),
            config,
            log_dir,
            log_sequence: AtomicUsize::new(0),
        }
    }

    /// Write a log entry to a file
    fn write_log_entry(&self, method: &str, entry: &LogEntry) {
        let Some(ref dir) = self.log_dir else {
            return;
        };

        let seq = self.log_sequence.fetch_add(1, Ordering::SeqCst);
        // Use timestamp with underscores instead of colons for filename compatibility
        let timestamp = entry.timestamp.replace(':', "-").replace('.', "-");
        let filename = format!("{}_{:03}_{}.json", timestamp, seq, method);
        let path = dir.join(&filename);

        match serde_json::to_string_pretty(entry) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    warn!("Failed to write log file {:?}: {}", path, e);
                }
            }
            Err(e) => {
                warn!("Failed to serialize log entry: {}", e);
            }
        }
    }

    /// Send a message to Claude and get a response
    pub async fn send_message(&self, system: &str, user: &str) -> Result<String> {
        let start = Instant::now();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

        let request = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            temperature: Some(self.config.temperature),
            system: Some(system.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: user.to_string(),
            }],
        };

        let request_json = serde_json::to_value(&request).unwrap_or_default();

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status_code = response.status().as_u16();
        let duration_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            let error_msg = format!("Anthropic API error: {} - {}", status_code, body);

            self.write_log_entry("send_message", &LogEntry {
                timestamp,
                method: "send_message".to_string(),
                duration_ms,
                request: request_json,
                response: None,
                status_code: Some(status_code),
                error: Some(error_msg.clone()),
            });

            anyhow::bail!(error_msg);
        }

        let response_bytes = response.bytes().await
            .context("Failed to read response bytes")?;
        let response_json: serde_json::Value = serde_json::from_slice(&response_bytes)
            .unwrap_or_default();

        self.write_log_entry("send_message", &LogEntry {
            timestamp,
            method: "send_message".to_string(),
            duration_ms,
            request: request_json,
            response: Some(response_json.clone()),
            status_code: Some(status_code),
            error: None,
        });

        let response: AnthropicResponse = serde_json::from_value(response_json)
            .context("Failed to parse Anthropic API response")?;

        // Extract text from the first content block
        response
            .content
            .first()
            .and_then(|c| {
                if c.content_type == "text" {
                    Some(c.text.clone())
                } else {
                    None
                }
            })
            .context("No text content in response")
    }

    /// Send a message with tool use for structured output
    pub async fn send_with_tool(&self, system: &str, user: &str) -> Result<(WindowPatch, Usage)> {
        let tool = Tool {
            name: "submit_patch".to_string(),
            description: "Submit the window patch with token relabels and turn edits".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "window_id": {
                        "type": "string",
                        "description": "ID of the window being patched"
                    },
                    "token_relabels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "token_id": {"type": "string"},
                                "new_speaker": {"type": "integer"},
                                "reason": {
                                    "type": "string",
                                    "enum": ["jitter_short_turn", "overlap_boundary", "lexical_continuity", "dialogue_pairing", "backchannel_attribution", "do_not_change"]
                                }
                            },
                            "required": ["token_id", "new_speaker", "reason"]
                        }
                    },
                    "turn_edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["merge_turns", "split_turn"]},
                                "turn_id": {"type": "string"},
                                "to_turn_id": {"type": "string"},
                                "split_at_token_id": {"type": "string"},
                                "reason": {
                                    "type": "string",
                                    "enum": ["jitter_short_turn", "overlap_boundary", "lexical_continuity", "dialogue_pairing", "backchannel_attribution", "do_not_change"]
                                }
                            },
                            "required": ["type", "turn_id", "reason"]
                        }
                    },
                    "violations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List any rules you may have violated"
                    },
                    "notes": {
                        "type": "object",
                        "properties": {
                            "uncertain_tokens": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "summary": {"type": "string"}
                        }
                    }
                },
                "required": ["window_id", "token_relabels", "turn_edits", "violations", "notes"]
            }),
        };

        let start = Instant::now();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

        let request = AnthropicToolRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            temperature: Some(self.config.temperature),
            system: Some(system.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: user.to_string(),
            }],
            tools: vec![tool],
            tool_choice: Some(ToolChoice {
                choice_type: "tool".to_string(),
                name: "submit_patch".to_string(),
            }),
        };

        let request_json = serde_json::to_value(&request).unwrap_or_default();

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status_code = response.status().as_u16();
        let duration_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            let error_msg = format!("Anthropic API error: {} - {}", status_code, body);

            self.write_log_entry("send_with_tool", &LogEntry {
                timestamp,
                method: "send_with_tool".to_string(),
                duration_ms,
                request: request_json,
                response: None,
                status_code: Some(status_code),
                error: Some(error_msg.clone()),
            });

            anyhow::bail!(error_msg);
        }

        let response_bytes = response.bytes().await
            .context("Failed to read response bytes")?;
        let response_json: serde_json::Value = serde_json::from_slice(&response_bytes)
            .unwrap_or_default();

        self.write_log_entry("send_with_tool", &LogEntry {
            timestamp,
            method: "send_with_tool".to_string(),
            duration_ms,
            request: request_json,
            response: Some(response_json.clone()),
            status_code: Some(status_code),
            error: None,
        });

        let response: AnthropicResponse = serde_json::from_value(response_json)
            .context("Failed to parse Anthropic API response")?;

        // Find the tool_use content block
        for content in &response.content {
            if content.content_type == "tool_use" && content.name.as_deref() == Some("submit_patch")
            {
                if let Some(input) = &content.input {
                    let patch: WindowPatch = serde_json::from_value(input.clone())
                        .context("Failed to parse tool input as WindowPatch")?;
                    return Ok((patch, response.usage));
                }
            }
        }

        anyhow::bail!("No tool_use response found")
    }

    /// Send a speaker identification request using tool use
    pub async fn send_speaker_id_request(
        &self,
        system: &str,
        user: &str,
    ) -> Result<(Vec<SpeakerIdentification>, Usage)> {
        let start = Instant::now();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

        let tool = Tool {
            name: "submit_speaker_identifications".to_string(),
            description: "Submit speaker identifications with confidence scores and evidence"
                .to_string(),
            input_schema: get_speaker_id_tool_schema(),
        };

        let request = AnthropicToolRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            temperature: Some(self.config.temperature),
            system: Some(system.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: user.to_string(),
            }],
            tools: vec![tool],
            tool_choice: Some(ToolChoice {
                choice_type: "tool".to_string(),
                name: "submit_speaker_identifications".to_string(),
            }),
        };

        let request_json = serde_json::to_value(&request).unwrap_or_default();

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status_code = response.status().as_u16();
        let duration_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            let error_msg = format!("Anthropic API error: {} - {}", status_code, body);

            self.write_log_entry("send_speaker_id_request", &LogEntry {
                timestamp,
                method: "send_speaker_id_request".to_string(),
                duration_ms,
                request: request_json,
                response: None,
                status_code: Some(status_code),
                error: Some(error_msg.clone()),
            });

            anyhow::bail!(error_msg);
        }

        let response_bytes = response.bytes().await
            .context("Failed to read response bytes")?;
        let response_json: serde_json::Value = serde_json::from_slice(&response_bytes)
            .unwrap_or_default();

        self.write_log_entry("send_speaker_id_request", &LogEntry {
            timestamp,
            method: "send_speaker_id_request".to_string(),
            duration_ms,
            request: request_json,
            response: Some(response_json.clone()),
            status_code: Some(status_code),
            error: None,
        });

        let response: AnthropicResponse = serde_json::from_value(response_json)
            .context("Failed to parse Anthropic API response")?;

        // Find the tool_use content block
        for content in &response.content {
            if content.content_type == "tool_use"
                && content.name.as_deref() == Some("submit_speaker_identifications")
            {
                if let Some(input) = &content.input {
                    let result: SpeakerIdToolResult = serde_json::from_value(input.clone())
                        .context("Failed to parse tool input as SpeakerIdToolResult")?;
                    return Ok((result.identifications, response.usage));
                }
            }
        }

        anyhow::bail!("No tool_use response found for speaker identification")
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
struct AnthropicToolRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct Tool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    name: String,
}

/// Token usage from API response
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

/// Internal struct for parsing speaker identification tool response
#[derive(Debug, Deserialize)]
struct SpeakerIdToolResult {
    identifications: Vec<SpeakerIdentification>,
}
