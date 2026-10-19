use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::RngExt as _;
use tokenizers::Tokenizer;

use crate::model;

pub fn run(prompt: &str, filtered: bool) -> anyhow::Result<()> {
	const CONTEXT_SIZE: usize = 256;
	const NEW_TOKENS: usize = 100;

	fn sample(logits: &Tensor, temperature: f64) -> candle_core::Result<u32> {
		use candle_core::D;
		use candle_nn::ops::softmax;

		let logits = (logits / temperature)?;
		let probs = softmax(&logits, D::Minus1)?.to_vec1::<f32>()?;

		let mut rng = rand::rng();
		let mut r = rng.random::<f64>();
		for (id, p) in probs.iter().enumerate() {
			r -= *p as f64;
			if r <= 0.0 {
				return Ok(id as u32);
			}
		}

		Ok((probs.len() - 1) as u32)
	}

	let device = Device::Cpu;
	let tokenizer = Tokenizer::from_file(format!("tokenizers/tokenizer_{}filtered.json", if filtered {
		""
	} else {
		"un"
	})).unwrap();

	let config = model::Config {
		vocab_size: tokenizer.get_vocab_size(true),
		context_size: CONTEXT_SIZE,
		..Default::default()
	};

	let mut vars = VarMap::new();
	let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
	let model = model::Gpt::new(config, vb)?;

	vars.load(format!("tensors/meow_{}filtered.safetensors", if filtered {
		""
	} else {
		"un"
	}))?;

	let eos = tokenizer.token_to_id("<|endoftext|>").unwrap();

	let mut tokens = vec![eos];
	tokens.extend_from_slice(tokenizer.encode(prompt, false).unwrap().get_ids());

	for _ in 0..NEW_TOKENS {
		let start = tokens.len().saturating_sub(CONTEXT_SIZE);
		let context = &tokens[start..];

		let input = Tensor::from_vec(
			context.to_vec(),
			(1, context.len()),
			&device,
		)?;

		let logits = model.forward(&input)?;
		let last = logits
			.narrow(1, context.len() - 1, 1)?
			.squeeze(1)?;

		let next = sample(&last.squeeze(0)?, 0.9)?;

		tokens.push(next);
	}

	let text = tokenizer
		.decode(&tokens, false)
		.unwrap()
		.replace("<|endoftext|>", " ");

	println!("{}", text.trim());

	Ok(())
}
