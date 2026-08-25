from qwen_moe_rmu.trainer import QwenMoeRMUTrainer


class GemmaMoeRMUTrainer(QwenMoeRMUTrainer):
    """RMU for Gemma 4, whose packed experts live directly below each layer."""

    def __init__(
        self,
        module_regex=r"model\.language_model\.layers\.9",
        trainable_params_regex=(
            r"model\.language_model\.layers\.(7|8|9)\.experts\."
            r"(gate_up_proj|down_proj)",
            r"model\.language_model\.layers\.(7|8|9)\.mlp\."
            r"(gate_proj|up_proj|down_proj)\.weight",
        ),
        *args,
        **kwargs,
    ):
        super().__init__(
            module_regex=module_regex,
            trainable_params_regex=trainable_params_regex,
            *args,
            **kwargs,
        )
