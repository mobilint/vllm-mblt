"""Registry metadata for independently compiled Mobilint pooling artifacts.

Execution lives in MbltPoolingWorker; these classes also provide vLLM's
cross-encoder prompt hook before requests reach that worker.
"""

from torch import nn

QWEN_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class MobilintEmbeddingModel(nn.Module):
    is_pooling_model = True
    # Attention is performed inside the MXQ, with no vLLM-managed KV cache.
    is_attention_free = True
    default_pooling_type = "MEAN"

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        raise RuntimeError("Load Mobilint pooling artifacts through MbltPoolingWorker")

    def embed_input_ids(self, input_ids):
        raise RuntimeError("Input embedding lookup is owned by MbltPoolingWorker")

    def forward(self, input_ids, positions):
        raise RuntimeError("Full-sequence MXQ execution is owned by MbltPoolingWorker")


class MobilintLastTokenEmbeddingModel(MobilintEmbeddingModel):
    default_pooling_type = "LAST"


class MobilintForSequenceClassification(MobilintEmbeddingModel):
    supports_cross_encoding = True
    default_pooling_type = "CLS"


class MobilintQwen3ForSequenceClassification(MobilintForSequenceClassification):
    supports_score_template = True
    default_pooling_type = "LAST"

    @classmethod
    def get_score_template(cls, query: str, document: str) -> str:
        return (
            f"{QWEN_RERANK_PREFIX}<Instruct>: {QWEN_RERANK_INSTRUCTION}\n"
            f"<Query>: {query}\n<Document>: {document}{QWEN_RERANK_SUFFIX}"
        )

    @classmethod
    def post_process_tokens(cls, prompt) -> None:
        # The suffix is already part of the template. Never append EOS after it.
        pass
