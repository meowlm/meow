use candle_core::{D, DType, Result, Tensor};
use candle_nn::{
    embedding, linear, Embedding, Linear, Module, VarBuilder,
};

pub struct Config {
    pub vocab_size: usize,
    pub context_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 8000,
            context_size: 256,
            hidden_size: 256,
            num_heads: 4,
            num_layers: 4,
            ff_size: 1024,
        }
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints(
                size,
                "weight",
                candle_nn::init::ONE,
            )?,
            bias: vb.get_with_hints(
                size,
                "bias",
                candle_nn::init::ZERO,
            )?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized =
            centered.broadcast_div(&variance.affine(1.0, self.eps)?.sqrt()?)?;

        normalized
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_size: usize,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;

        assert_eq!(
            hidden % config.num_heads,
            0,
            "hidden_size must be divisible by num_heads"
        );

        Ok(Self {
            q: linear(hidden, hidden, vb.pp("q"))?,
            k: linear(hidden, hidden, vb.pp("k"))?,
            v: linear(hidden, hidden, vb.pp("v"))?,
            out: linear(hidden, hidden, vb.pp("out"))?,
            heads: config.num_heads,
            head_size: hidden / config.num_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, hidden) = x.dims3()?;

        let reshape_heads = |x: Tensor| -> Result<Tensor> {
            x.reshape((batch, seq, self.heads, self.head_size))?
                .transpose(1, 2)
        };

        let q = reshape_heads(self.q.forward(x)?)?;
        let k = reshape_heads(self.k.forward(x)?)?;
        let v = reshape_heads(self.v.forward(x)?)?;

        let scores = q
			.contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_size as f64).sqrt(), 0.0)?;

        let future = Tensor::triu2(seq, DType::F32, x.device())?
            .sub(&Tensor::eye(seq, DType::F32, x.device())?)?;

        let scores = scores.broadcast_add(&future.affine(-1e9, 0.0)?)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights.contiguous()?.matmul(&v.contiguous()?)?;

        attended
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, hidden))
            .and_then(|x| self.out.forward(&x))
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            up: linear(
                config.hidden_size,
                config.ff_size,
                vb.pp("up"),
            )?,
            down: linear(
                config.ff_size,
                config.hidden_size,
                vb.pp("down"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(x)?.gelu()?)
    }
}

struct Block {
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    ff: FeedForward,
}

impl Block {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(
                config.hidden_size,
                vb.pp("norm1"),
            )?,
            attention: Attention::new(config, vb.pp("attention"))?,
            norm2: LayerNorm::new(
                config.hidden_size,
                vb.pp("norm2"),
            )?,
            ff: FeedForward::new(config, vb.pp("ff"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.attention.forward(&self.norm1.forward(x)?)?)?;
        let ff = self.ff.forward(&self.norm2.forward(&x)?)?;
        x.add(&ff)
    }
}

pub struct Gpt {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    config: Config,
}

impl Gpt {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let token_embedding = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("token_embedding"),
        )?;

        let position_embedding = embedding(
            config.context_size,
            config.hidden_size,
            vb.pp("position_embedding"),
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            blocks.push(Block::new(&config, vb.pp(format!("blocks.{i}")))?);
        }

        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            norm: LayerNorm::new(config.hidden_size, vb.pp("norm"))?,
            head: linear(
                config.hidden_size,
                config.vocab_size,
                vb.pp("head"),
            )?,
            config,
        })
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, seq) = tokens.dims2()?;

        if seq > self.config.context_size {
            candle_core::bail!("Sequence exceeds context size");
        }

        let positions =
            Tensor::arange(0u32, seq as u32, tokens.device())?;

        let token_embeddings = self.token_embedding.forward(tokens)?;
        let position_embeddings = self
            .position_embedding
            .forward(&positions)?
            .unsqueeze(0)?
            .broadcast_as((batch, seq, self.config.hidden_size))?;

        let mut x = token_embeddings.add(&position_embeddings)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        self.head.forward(&self.norm.forward(&x)?)
    }
}
