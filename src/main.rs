mod tokenizer;
mod dataset;
mod model;
mod train;
mod run;

use clap::Parser as _;

#[derive(clap::Parser, Debug)]
#[clap(name = "meow", version = "0.0.1")]
pub struct App {
    #[clap(subcommand)]
    pub command: Subcommand,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum Subcommand {
	#[command(about = "train the model")]
	Train {
		#[arg(help = "whether to train the filtered model or the unfiltered model", default_value_t = true)]
		filtered: bool
	},

	#[command(about = "invoke the model directly, provided its trained already")]
	Run {
		#[arg(help = "what prompt to give to the llm", default_value = "mrrr", action = clap::ArgAction::Set)]
		prompt: String,

		#[arg(help = "whether to use the filtered model or the unfiltered model", default_value_t = true, action = clap::ArgAction::Set)]
		filtered: bool
	},

	#[command(about = "generate dataset from exported discord channel")]
	Gen,
}

fn main() -> anyhow::Result<()> {
	let app = App::parse();
	
	match app.command {
		Subcommand::Train { filtered } => train::train(filtered)?,
		Subcommand::Run { prompt, filtered } => run::run(prompt.as_str(), filtered)?,
		Subcommand::Gen => dataset::r#gen()?
	};

	Ok(())
}
