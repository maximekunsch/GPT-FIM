import torch
from transformers import PreTrainedModel, PretrainedConfig
from Model import GPT, GPTConfig

from transformers import GPT2TokenizerFast
import os

class HFConfig(PretrainedConfig):
    model_type = "custom-gpt-fim"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HFGPT(PreTrainedModel):
    config_class = HFConfig
    
    def __init__(self, config):
        super().__init__(config)
        gpt_cfg = GPTConfig(
            block_size=config.block_size,
            sliding_window=config.sliding_window,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            softcap=config.softcap,
            dropout=0,
            bias=False,
        )
        self.model = GPT(gpt_cfg)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits, loss = self.model(input_ids, labels)
        return {"logits": logits, "loss": loss}
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


cfg = HFConfig(
    block_size=1024,
    sliding_window=512,
    vocab_size=100277,
    n_layer=9,
    n_head=16,
    n_embd=2048,
    softcap=20,
)

model = HFGPT(cfg)
sd = torch.load("gpt_model.pt", map_location="cpu")
model.model.load_state_dict(sd, strict=True)

model.save_pretrained("my_fim_gpt", safe_serialization=False)







tok = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4")

tok.add_special_tokens({
    "additional_special_tokens": [
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"
    ]
})
tok.save_pretrained("my_fim_gpt")



