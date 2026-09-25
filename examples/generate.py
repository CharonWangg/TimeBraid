"""Run one greedy TimeBraid text-and-time-series generation request."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Local TimeBraid artifact directory or Hub ID",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TimeBraid mixed time-series inference requires CUDA.")
    import flash_attn  # noqa: F401 - fail before loading model weights
    from transformers import AutoModelForCausalLM, AutoProcessor

    from timebraid import TimeBraidProcessor  # noqa: F401 - registers HF AutoClasses

    processor = AutoProcessor.from_pretrained(
        args.model,
        fix_mistral_regex=False,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        trust_remote_code=False,
        use_safetensors=True,
    ).eval()
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": "Forecast the next 8 values."}],
        timeseries=[[101.2, 101.8, 102.1, 102.5, 103.0, 103.4]],
        horizon=8,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    result = processor.post_process_generation(outputs, model_inputs=inputs)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
