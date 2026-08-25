"""Client for a single reference model hosted on a dedicated GPU."""

from multiprocessing import current_process
from multiprocessing.connection import Client

import torch


class ReferenceClient:
    def __init__(self, host, port, authkey="open-unlearning"):
        self.address = (str(host), int(port))
        self.authkey = str(authkey).encode("utf-8")
        # Torch tensors are transferred through multiprocessing's resource
        # sharer, which authenticates independently from the main Connection.
        current_process().authkey = self.authkey

    @staticmethod
    def _cpu_inputs(inputs):
        return {
            key: value.detach().to("cpu") if torch.is_tensor(value) else value
            for key, value in inputs.items()
            if key in ("input_ids", "attention_mask", "labels")
        }

    def _request(self, operation, inputs):
        connection = Client(self.address, authkey=self.authkey)
        try:
            connection.send({"operation": operation, "inputs": self._cpu_inputs(inputs)})
            response = connection.recv()
        finally:
            connection.close()
        if not response.get("ok"):
            raise RuntimeError(f"Reference server failed: {response.get('error')}")
        return response["value"]

    def sequence_nll(self, inputs):
        return self._request("sequence_nll", inputs)

    def logits(self, inputs):
        return self._request("logits", inputs)
