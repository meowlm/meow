use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer as _, ParamsAdamW, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{dataset, model, tokenizer};

#[derive(Deserialize)]
struct Sample {
    text: String,
}

const SEQ_LEN: usize = 256;
const BATCH_SIZE: usize = 4;
const STEPS: usize = 10_000;
const LEARNING_RATE: f64 = 3e-4;

pub fn train(filtered: bool) -> anyhow::Result<()> {
    let file = std::fs::read_to_string(format!("data/meow_{}filtered.jsonl", if filtered {
		""
	} else {
		"un"
	}))?;

	let samples = file
        .lines()
        .map(|line| {
            serde_json::from_str::<Sample>(&line).unwrap()
        }).collect::<Vec<_>>();

    println!("loaded {} meows", samples.len());

    if let Some(sample) = samples.first() {
        println!("first one: {}", sample.text.chars().take(200).collect::<String>());
    }

	let texts = samples
		.iter().map(|sample| sample.text.as_str());

	let tokenizer_path = format!("tokenizers/tokenizer_{}filtered.json", if filtered {
		""
	} else {
		"un"
	});

	tokenizer::train(texts.clone().collect(), &tokenizer_path)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let tokens = dataset::load(texts.collect(), &tokenizer).unwrap();

	anyhow::ensure!(
		tokens.len() > SEQ_LEN + 1,
		"dataset needs more than {} tokens; got {}",
		SEQ_LEN + 1,
		tokens.len()
	);

	let device = candle_core::Device::Cpu;

	let config = model::Config {
		vocab_size: tokenizer.get_vocab_size(true),
		context_size: SEQ_LEN,
		..Default::default()
	};

	let vars = VarMap::new();
	let vb = candle_nn::VarBuilder::from_varmap(&vars, candle_core::DType::F32, &device);
	let model = model::Gpt::new(config, vb)?;

	let mut optimizer = AdamW::new(
		vars.all_vars(),
		ParamsAdamW {
			lr: LEARNING_RATE,
			..Default::default()
		},
	)?;

	let prog = ProgressBar::new(STEPS as u64);
	
    prog.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.purple} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("=>-"));

	for step in 0..STEPS {
		let mut inputs = Vec::with_capacity(BATCH_SIZE * SEQ_LEN);
		let mut targets = Vec::with_capacity(BATCH_SIZE * SEQ_LEN);

		for batch in 0..BATCH_SIZE {
			let start = (step * BATCH_SIZE + batch) % (tokens.len() - SEQ_LEN - 1);

			inputs.extend_from_slice(&tokens[start..start + SEQ_LEN]);
			targets.extend_from_slice(&tokens[start + 1..start + SEQ_LEN + 1]);
		}

		let input = Tensor::from_vec(
			inputs,
			(BATCH_SIZE, SEQ_LEN),
			&device,
		)?;

		let target = Tensor::from_vec(
			targets,
			(BATCH_SIZE * SEQ_LEN,),
			&device,
		)?
		.to_dtype(candle_core::DType::U32)?;

		let logits = model.forward(&input)?
			.reshape((BATCH_SIZE * SEQ_LEN, tokenizer.get_vocab_size(true)))?;

		let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
		let loss = candle_nn::loss::nll(&log_probs, &target)?;

		optimizer.backward_step(&loss)?;

		prog.set_message(format!("step {step}: loss = {:.4}", loss.to_scalar::<f32>()?));

		prog.inc(1);
	}

	vars.save(format!("tensors/meow_{}filtered.safetensors", if filtered {
		""
	} else {
		"un"
	}))?;

    println!("{} tokens", tokens.len());
    println!("vocab size: {}", tokenizer.get_vocab_size(true));

    Ok(())
}

