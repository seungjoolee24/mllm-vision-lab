"""BF16/FP16 Qwen inference with explicit resolution and memory accounting."""
from contextlib import contextmanager
import gc
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import time
from .data import make_prompt, sha256_file
from .core import digest


def runtime_info(dtype="auto"):
    import torch
    available = torch.cuda.is_available()
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": available,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("transformers", "torchvision", "accelerate", "huggingface-hub", "pillow",
                          "tokenizers", "numpy", "regex", "safetensors")}}
    info["installed_distributions"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                        if d.metadata.get("Name")}
    if available:
        free, total = torch.cuda.mem_get_info()
        chosen = "bfloat16" if dtype == "auto" and torch.cuda.is_bf16_supported() else "float16" if dtype == "auto" else dtype
        info.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                    vram_gib=round(total / 2**30, 3), free_gib=round(free / 2**30, 3),
                    bf16_supported=torch.cuda.is_bf16_supported(), effective_dtype=chosen)
    try:
        info["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                     "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi"] = None
    return info


def extract_boxed(text):
    import regex
    matches = regex.findall(r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}', text)
    if matches:
        return matches[-1].strip()
    matches = regex.findall(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', text)
    return matches[-1].strip() if matches else None


class QwenBackend:
    def __init__(self, config, dataset, model_path=None):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from transformers.models.qwen2_vl import image_processing_qwen2_vl_fast
        self.config, self.dataset = config, Path(dataset)
        self.runtime = runtime_info(config["dtype"])
        if not self.runtime["cuda_available"]:
            raise RuntimeError("A CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU.")
        if self.runtime["effective_dtype"] == "bfloat16" and not self.runtime["bf16_supported"]:
            raise RuntimeError("This GPU does not support the requested BF16 condition. Select an L4/A100-class "
                               "BF16 GPU, or explicitly use --dtype float16 and a NEW run-id.")
        if transformers.__version__ != "4.57.6":
            raise RuntimeError("The controlled resize wrapper requires transformers==4.57.6")
        self.runtime_key = {k: self.runtime.get(k) for k in
                            ("gpu", "capability", "effective_dtype", "torch", "cuda_runtime", "packages")}
        self.runtime_key["model_identity"] = {"id": config["model_id"], "revision": config["model_revision"]}
        if model_path:
            files = sorted(Path(model_path).glob("*.safetensors")) + sorted(Path(model_path).glob("*.json"))
            if not files:
                raise ValueError("Local model directory has no weights/configuration")
            self.runtime_key["local_model_sha256"] = digest({p.name: sha256_file(p) for p in files})
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        free, total = torch.cuda.mem_get_info()
        limit = min(total * config.get("allocator_fraction", 0.9), free - 256 * 1024**2)
        if limit <= 0:
            raise RuntimeError("Insufficient free VRAM")
        torch.cuda.set_per_process_memory_fraction(limit / total)
        self.runtime["allocator_limit_gib"] = round(limit / 2**30, 3)
        source = str(model_path) if model_path else config["model_id"]
        revision = {} if model_path else {"revision": config["model_revision"]}
        self.processor = AutoProcessor.from_pretrained(source, **revision,
            min_pixels=config["min_visual_tokens"] * 32**2, max_pixels=config["max_visual_tokens"] * 32**2)
        self.processor.tokenizer.padding_side = "left"
        self.image_module = image_processing_qwen2_vl_fast
        self.original_resize = self.image_module.smart_resize
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(source, **revision,
            dtype=getattr(torch, self.runtime["effective_dtype"]), device_map={"": 0},
            attn_implementation="sdpa").eval()
        if {str(p.device) for p in self.model.parameters()} != {"cuda:0"}:
            raise RuntimeError("Unexpected CPU/offloaded model parameters")
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    @contextmanager
    def scale(self, factor):
        def resized(height, width, *args, **kwargs):
            h, w = self.original_resize(height, width, *args, **kwargs)
            return max(32, round(h * factor / 32) * 32), max(32, round(w * factor / 32) * 32)
        self.image_module.smart_resize = resized
        try:
            yield
        finally:
            self.image_module.smart_resize = self.original_resize

    def infer(self, questions, condition, seed):
        import torch
        from PIL import Image
        images, texts, prompts = [], [], []
        for q in questions:
            # The inference boundary accepts input fields only, never an answer or reference CoT.
            if any(k in q for k in ("ground_truth", "choiceAns", "blankAns", "coT")):
                raise ValueError("Answer key reached inference boundary")
            path = self.dataset / q["image"]
            if sha256_file(path) != q["image_sha256"]:
                raise ValueError(f"Image changed for task {q['taskId']}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            prompt = make_prompt(q, self.config["prompt_suffix"])
            prompts.append(prompt)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        with self.scale(condition["spatial_scale"]):
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"],
                                          **self.config["sampling"], use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        generated = outputs[:, inputs.input_ids.shape[1]:]
        responses = self.processor.batch_decode(generated, skip_special_tokens=True)
        records = []
        for i, q in enumerate(questions):
            tokens = generated[i].tolist()
            stop = next((j for j, token in enumerate(tokens) if token in self.eos), None)
            length = stop + 1 if stop is not None else len(tokens)
            grid = inputs.image_grid_thw[i].tolist()
            records.append({"task_id": q["taskId"], "prompt": prompts[i], "response": responses[i].strip(),
                "extracted_answer": extract_boxed(responses[i]), "input_tokens": int(inputs.attention_mask[i].sum()),
                "image_grid_thw": grid, "visual_tokens": grid[0] * grid[1] * grid[2] // 4,
                "original_image_size": list(images[i].size), "processed_image_size": [grid[2]*16, grid[1]*16],
                "output_tokens": length, "hit_token_limit": stop is None and length >= self.config["max_new_tokens"],
                "peak_allocated_gib": peak, "batch_seconds": round(elapsed, 3),
                "generation_seed": seed, "batch_task_ids": [r["taskId"] for r in questions]})
        return records

    def attempt(self, questions, condition, seed):
        import torch
        try:
            return self.infer(questions, condition, seed), None
        except torch.cuda.OutOfMemoryError as error:
            message = str(error)
        # Release the traceback's tensor references before empty_cache.
        gc.collect()
        torch.cuda.empty_cache()
        return None, message
