"""Host exactly one frozen reference model on a dedicated GPU.

The server deliberately returns CPU tensors. Sequence NPO/DPO request
per-sequence NLLs, TokenNPO requests per-token NLLs, and exact KL/UNDIAL request
logits and are consequently bandwidth-heavy.
"""

import argparse
import os
import threading
from multiprocessing import current_process
from multiprocessing.connection import Listener

import torch
import torch.nn.functional as F


def load_model(model_path, model_handler, dtype):
    import transformers

    cls = getattr(transformers, model_handler, None)
    if cls is None:
        raise RuntimeError(f"transformers.{model_handler} is unavailable")
    torch_dtype = getattr(torch, dtype)
    model = cls.from_pretrained(
        model_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def sequence_nll(logits, labels):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.float().transpose(-1, -2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    ).sum(dim=-1)


def token_nll(logits, labels):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.float().transpose(-1, -2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-handler", default="AutoModelForMultimodalLM")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29591)
    parser.add_argument("--authkey", default="open-unlearning")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ready-file")
    args = parser.parse_args()

    # Match the authkey used by PyTorch's tensor resource-sharing side channel.
    current_process().authkey = args.authkey.encode("utf-8")

    model = load_model(args.model_path, args.model_handler, args.dtype)
    lock = threading.Lock()
    listener = Listener((args.host, args.port), authkey=args.authkey.encode("utf-8"))
    if args.ready_file:
        with open(args.ready_file, "w", encoding="utf-8") as handle:
            handle.write(f"{args.host}:{args.port}\n")

    def serve(connection):
        try:
            request = connection.recv()
            inputs = {
                key: value.to("cuda:0", non_blocking=True)
                for key, value in request["inputs"].items()
            }
            with lock, torch.inference_mode():
                outputs = model(**inputs)
                if request["operation"] == "sequence_nll":
                    value = sequence_nll(outputs.logits, inputs["labels"])
                elif request["operation"] == "token_nll":
                    value = token_nll(outputs.logits, inputs["labels"])
                elif request["operation"] == "logits":
                    value = outputs.logits
                else:
                    raise ValueError(f"Unknown operation: {request['operation']}")
                value = value.detach().to("cpu")
            connection.send({"ok": True, "value": value})
        # This is an RPC boundary: serialize any model/runtime failure back to
        # the caller instead of terminating the long-lived reference server.
        except Exception as exc:  # noqa: BLE001
            connection.send({"ok": False, "error": repr(exc)})
        finally:
            connection.close()

    try:
        while True:
            connection = listener.accept()
            threading.Thread(target=serve, args=(connection,), daemon=True).start()
    finally:
        listener.close()
        if args.ready_file and os.path.exists(args.ready_file):
            os.unlink(args.ready_file)


if __name__ == "__main__":
    main()
