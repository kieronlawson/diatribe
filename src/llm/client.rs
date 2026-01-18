use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::models::WindowPatch;

/// Configuration for the Anthropic API client
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key (from ANTHROPIC_API_KEY env var)
    pub api_key: String,
    /// Model to use (e.g., "claude-sonnet-4-20250514")
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
            model: "claude-sonnet-4-20250514".to_string(),
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

/// Anthropic API client
pub struct AnthropicClient {
    client: Client,
    config: AnthropicConfig,
}

impl AnthropicClient {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Send a message to Claude and get a response
    pub async fn send_message(&self, system: &str, user: &str) -> Result<String> {
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

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error: {} - {}", status, body);
        }

        let response: AnthropicResponse = response
            .json()
            .await
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
    pub async fn send_with_tool(&self, system: &str, user: &str) -> Result<WindowPatch> {
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

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error: {} - {}", status, body);
        }

        let response: AnthropicResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic API response")?;

        // Find the tool_use content block
        for content in &response.content {
            if content.content_type == "tool_use" && content.name.as_deref() == Some("submit_patch")
            {
                if let Some(input) = &content.input {
                    let patch: WindowPatch = serde_json::from_value(input.clone())
                        .context("Failed to parse tool input as WindowPatch")?;
                    return Ok(patch);
                }
            }
        }

        anyhow::bail!("No tool_use response found")
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

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
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
