"""vLLM V1 worker for full-sequence, unpadded Mobilint pooling inference."""

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from vllm_mblt.pooling import PoolingRuntime


class MbltPoolingWorker(WorkerBase):
    def init_device(self):
        self.model = None
        self.requests = {}

    def load_model(self):
        extra = self.load_config.model_loader_extra_config or {}
        unknown = set(extra) - {"dev_no", "core"}
        if unknown:
            raise ValueError(f"Unsupported pooling runtime options: {sorted(unknown)}")
        self.model = PoolingRuntime(self.model_config.model, revision=self.model_config.revision, **extra)
        try:
            config = self.model_config.hf_config
            if config.mblt_pooling != self.model.kind or config.hidden_size != self.model.hidden_size:
                raise ValueError("vLLM pooling task/dimensions do not match the compiled package")
            if self.model_config.is_matryoshka and not self.model.manifest.get("matryoshka", False):
                raise ValueError("vLLM advertises dimension reduction unsupported by the compiled package")
            if self.model_config.max_model_len > self.model.max_length:
                raise ValueError("max_model_len exceeds the compiled pooling artifact's sequence limit")
        except Exception:
            self.model.close()
            self.model = None
            raise

    def get_supported_tasks(self):
        kind = self.model_config.hf_config.mblt_pooling
        return ("embed",) if kind in ("mean", "last") else ("classify", "score")

    def get_kv_cache_spec(self):
        return {}

    def determine_available_memory(self):
        return 0

    def initialize_from_config(self, kv_cache_config):
        pass

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
        pass

    def compile_or_warm_up_model(self):
        pass

    def get_model(self):
        return self.model

    def execute_model(self, scheduler_output):
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        for req in scheduler_output.scheduled_new_reqs:
            if req.prompt_embeds is not None or req.mm_features or req.lora_request is not None:
                raise ValueError(
                    "Pooling MXQs accept text token IDs only; embeddings, multimodal and LoRA are unsupported"
                )
            if req.pooling_params is None or req.num_computed_tokens:
                raise ValueError("Pooling requires a complete uncached prompt and pooling parameters")
            self.requests[req.req_id] = req
        ids = list(scheduler_output.num_scheduled_tokens)
        outputs = []
        for req_id in ids:
            req = self.requests[req_id]
            if scheduler_output.num_scheduled_tokens[req_id] != len(req.prompt_token_ids):
                raise ValueError("Pooling MXQs require full-sequence scheduling, not chunked prefill")
            outputs.append(self.model.encode(req.prompt_token_ids, req.pooling_params))
            del self.requests[req_id]
        return ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=outputs,
        )

    def check_health(self):
        if self.model is None:
            raise RuntimeError("Pooling runtime has not been loaded")

    def shutdown(self):
        if self.model is not None:
            self.model.close()
        self.requests.clear()
