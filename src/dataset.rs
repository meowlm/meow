use tokenizers::Tokenizer;
use std::{fs::File, io::{BufWriter, Write}};

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportedThread {
	pub messages: Vec<ExportedMessage>,
	pub message_count: i32
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct ExportedMessage {
	pub id: String,
	pub content: String,
	pub author: ExportedAuthor
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct ExportedAuthor {
	pub name: String,
	pub nickname: String
}

pub fn load(texts: Vec<&str>, tokenizer: &Tokenizer) -> anyhow::Result<Vec<u32>> {
	let eos = tokenizer
		.token_to_id("<|endoftext|>")
		.ok_or_else(|| anyhow::anyhow!("tokenizer is missing <|endoftext|>"))?;

	let mut tokens = vec![];

	for text in texts {
		let enc = tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!(e))?;
		tokens.extend_from_slice(enc.get_ids());
		tokens.push(eos);
	}

	Ok(tokens)
}

pub fn r#gen() -> anyhow::Result<()> {
	let meows = std::fs::read_to_string("data/meow.json").unwrap();

	let thread: ExportedThread = serde_json::from_str(&meows).unwrap();

	println!("found {} messages", thread.message_count);

	let messages = thread.messages.iter().map(|x| x.content.as_str()).collect::<Vec<_>>();

	let mut writer = BufWriter::new(File::create("data/meow_unfiltered.jsonl").unwrap());

	for text in &messages {
		serde_json::to_writer(&mut writer, &serde_json::json!({
			"text": text
		})).unwrap();
		writer.write_all(b"\n").unwrap();
	}

	writer.flush().unwrap();

    Ok(())
}
