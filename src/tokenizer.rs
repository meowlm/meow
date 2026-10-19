use tokenizers::{AddedToken, Tokenizer, decoders::byte_level::ByteLevel, models::{TrainerWrapper, bpe::{BPE, BpeTrainer}}, pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer};

pub fn train(dataset: Vec<&str>, output: &str) -> anyhow::Result<()> {
	let bpe = BPE::default();
	let mut tokenizer: Tokenizer = Tokenizer::new(bpe).into();

	tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::new(false, false, true)));

	tokenizer.with_decoder(Some(ByteLevel::default()));

	let mut trainer: TrainerWrapper = BpeTrainer::builder()
		.vocab_size(8000)
		.special_tokens(vec![AddedToken::from("<|endoftext|>", true)])
		.show_progress(true)
		.build()
		.into();

	tokenizer.train(&mut trainer, dataset.iter()).unwrap();
	tokenizer.save(output, true).unwrap();

	Ok(())
}
