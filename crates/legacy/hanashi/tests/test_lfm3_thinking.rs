use std::sync::Arc;

use hanashi::{
    Encoding as _,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::{
    basic::{ReasoningEffort, ToolDescription, ToolFunction, ToolNamespace},
    session::chat::{ChatContentBlock, ChatMessage},
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

fn encoding(config: HanashiConfig) -> HanashiEncodingImpl {
    let mut config = config.resolve().unwrap();
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocabulary: [(String, u32); 256] = alphabet
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value.to_string(), index as u32))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let model = BPE::builder().vocab_and_merges(vocabulary, Vec::new()).build().unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.add_tokens(&["system", "user", "assistant", "tool"].map(|value| AddedToken::from(value, false)));
    let mut markers = config.parsing.framing_config().tokens;
    markers.extend(["<bos>".to_string(), "<eos>".to_string()]);
    tokenizer.add_special_tokens(&markers.into_iter().map(|value| AddedToken::from(value, true)).collect::<Vec<_>>());
    config.tokens.bos_token_id = tokenizer.token_to_id("<bos>");
    config.tokens.eos_token_id = tokenizer.token_to_id("<eos>");
    HanashiEncodingImpl::new(
        HanashiConfig::Custom {
            config,
        },
        Arc::new(tokenizer),
    )
    .unwrap()
}

fn reply(
    reasoning: &str,
    text: &str,
) -> ChatMessage {
    let mut message = ChatMessage::assistant();
    if !reasoning.is_empty() {
        message = message.with_reasoning(reasoning.to_string());
    }
    if !text.is_empty() {
        message = message.with_text(text.to_string());
    }
    message
}

#[test]
fn lfm3_replays_reasoning_from_the_latest_assistant_turn() {
    for config in [HanashiConfig::Lfm3, HanashiConfig::Lfm3Vl] {
        let mut encoding = encoding(config);
        encoding
            .encode(vec![
                ChatMessage::user().with_text("first question".into()),
                reply("thinking deeply", "first answer"),
                ChatMessage::user().with_text("second question".into()),
            ])
            .unwrap();
        let prompt = encoding.state().text();
        assert!(
            prompt.contains("<|im_start|>assistant\n<think>thinking deeply</think>first answer<|im_end|>\n"),
            "{prompt:?}"
        );
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>"), "{prompt:?}");
    }
}

#[test]
fn lfm3_strips_reasoning_from_older_assistant_turns() {
    let mut encoding = encoding(HanashiConfig::Lfm3Vl);
    encoding
        .encode(vec![
            ChatMessage::user().with_text("one".into()),
            reply("older thoughts", "older answer"),
            ChatMessage::user().with_text("two".into()),
            reply("newer thoughts", "newer answer"),
            ChatMessage::user().with_text("three".into()),
        ])
        .unwrap();
    let prompt = encoding.state().text();
    assert!(prompt.contains("<|im_start|>assistant\n<think></think>older answer<|im_end|>\n"), "{prompt:?}");
    assert!(
        prompt.contains("<|im_start|>assistant\n<think>newer thoughts</think>newer answer<|im_end|>\n"),
        "{prompt:?}"
    );
}

#[test]
fn lfm3_renders_reasoning_only_assistant_turn() {
    let mut encoding = encoding(HanashiConfig::Lfm3Vl);
    encoding
        .encode(vec![
            ChatMessage::user().with_text("one".into()),
            reply("only thoughts", ""),
            ChatMessage::user().with_text("two".into()),
        ])
        .unwrap();
    let prompt = encoding.state().text();
    assert!(prompt.contains("<|im_start|>assistant\n<think>only thoughts</think><|im_end|>\n"), "{prompt:?}");
}

#[test]
fn lfm3_supports_disabling_thinking() {
    for config in [HanashiConfig::Lfm3, HanashiConfig::Lfm3Vl] {
        let capabilities = config.capabilities().unwrap();
        assert!(capabilities.supports_reasoning, "{config:?}");
        assert!(capabilities.supports_disable_reasoning, "{config:?}");
        assert!(capabilities.reasoning_efforts.contains(&ReasoningEffort::Disabled), "{config:?}");
        assert!(capabilities.reasoning_efforts.contains(&ReasoningEffort::Default), "{config:?}");
        assert_eq!(config.default_reasoning_effort().unwrap(), Some(ReasoningEffort::Default), "{config:?}");
    }
}

#[test]
fn lfm3_disabled_thinking_emits_closed_think_prefix() {
    for config in [HanashiConfig::Lfm3, HanashiConfig::Lfm3Vl] {
        let mut enc = encoding(config.clone());
        enc.encode(vec![
            ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
            ChatMessage::user().with_text("hello".into()),
        ])
        .unwrap();
        let prompt = enc.state().text();
        assert!(!prompt.contains("<|im_start|>system\n"), "{prompt:?}");
        assert!(prompt.ends_with("<|im_start|>assistant\n<think></think>"), "{prompt:?}");

        let mut enc = encoding(config.clone());
        enc.encode(vec![
            ChatMessage::system().with_text("Be concise.".into()).with_reasoning_effort(ReasoningEffort::Disabled),
            ChatMessage::user().with_text("hello".into()),
        ])
        .unwrap();
        let prompt = enc.state().text();
        assert!(prompt.contains("<|im_start|>system\nBe concise.<|im_end|>\n"), "{prompt:?}");
        assert!(prompt.ends_with("<|im_start|>assistant\n<think></think>"), "{prompt:?}");

        let mut enc = encoding(config);
        enc.encode(vec![
            ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default),
            ChatMessage::user().with_text("hello".into()),
        ])
        .unwrap();
        let prompt = enc.state().text();
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>"), "{prompt:?}");
    }
}

#[test]
fn lfm3_deferred_tools_and_tool_references() {
    let search_tool = ToolDescription::Function {
        tool_function: ToolFunction {
            name: "search_tools".to_string(),
            description: "Search for tools".to_string(),
            parameters: None,
            return_definition: None,
        },
    };
    let weather_tool = ToolDescription::Function {
        tool_function: ToolFunction {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: None,
            return_definition: None,
        },
    };
    let mut enc = encoding(HanashiConfig::Lfm3);
    enc.encode(vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::developer().with_tool_namespaces(vec![
            ToolNamespace {
                name: "functions".to_string(),
                description: None,
                tools: vec![search_tool],
            },
            ToolNamespace {
                name: "deferred".to_string(),
                description: None,
                tools: vec![weather_tool],
            },
        ]),
        ChatMessage::user().with_text("What is the weather?".into()),
        ChatMessage::assistant().with_text("Searching tools".into()),
        ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: Some("call_1".to_string()),
            name: Some("search_tools".to_string()),
            value: serde_json::json!([
                {"type": "text", "text": "Found 1 tool."},
                {"type": "tool_reference", "name": "get_weather"}
            ])
            .into(),
        }),
    ])
    .unwrap();
    let prompt = enc.state().text();
    let system_section = prompt.split("<|im_end|>").next().unwrap();
    assert!(system_section.contains("\"name\": \"search_tools\""), "{system_section:?}");
    assert!(!system_section.contains("get_weather"), "{system_section:?}");
    assert!(!system_section.contains("defer_loading"), "{system_section:?}");
    assert!(
        prompt.contains(
            "<|im_start|>tool\n<tool_response>\nFound 1 tool.\n<additional_tools>\n{\"name\": \"get_weather\", \"description\": \"Get weather\"}\n</additional_tools>\n</tool_response><|im_end|>\n"
        ),
        "{prompt:?}"
    );
    assert!(prompt.ends_with("<|im_start|>assistant\n<think></think>"), "{prompt:?}");
}
