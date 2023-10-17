import torch
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from decoding import infer
import json

from bayes_opt import BayesianOptimization


class LayerSkippingSearching:
    def __init__(
        self,
        model,
        tokenizer,
        evaluate_prompts,
        evaluate_config={"generate_fn": "essg", "max_new_tokens": 32},
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.evaluate_prompts = evaluate_prompts
        self.evaluate_config = evaluate_config

        self.pbounds = {
            f"x{i}": (0, 1) for i in range(self.config.num_hidden_layers * 2)
        }

        self.optimizer = BayesianOptimization(
            f=self._black_box_evaluate_function, pbounds=self.pbounds, random_state=1, verbose=1
        )

        self.optimizer.set_gp_params(alpha=1e-2)

    def _black_box_evaluate_function(self, **kargs):
        attn_skip_layers = []
        for i in range(self.config.num_hidden_layers):
            if kargs[f"x{i}"] > 0.5:
                attn_skip_layers.append(i)
        mlp_skip_layers = []
        for i in range(
            self.config.num_hidden_layers, self.config.num_hidden_layers * 2
        ):
            if kargs[f"x{i}"] > 0.5:
                mlp_skip_layers.append(i - self.config.num_hidden_layers)

        self.model.set_skip_layers(
            attn_skip_layer_id_set=attn_skip_layers,
            mlp_skip_layer_id_set=mlp_skip_layers,
        )

        total_time = 0
        total_tokens = 0

        for prompt in self.evaluate_prompts:
            ret = infer(self.model, self.tokenizer, prompt, **self.evaluate_config)
            total_time += ret["time"]
            total_tokens += self.evaluate_config.get("max_new_tokens", 10)

        print(
            "Log:",
            total_tokens / total_time,
            "tokens/s",
            "Skipped attn:",
            len(attn_skip_layers),
            "Skipped mlp:",
            len(mlp_skip_layers),
        )

        return total_tokens / total_time

    def probe(self, attn_skip_layers, mlp_skip_layers):
        """
        Add some good points to accelerate searching
        """

        params = {f"x{i}": 0.0 for i in range(self.config.num_hidden_layers * 2)}
        for i in attn_skip_layers:
            params[f"x{i}"] = 1.0
        for i in mlp_skip_layers:
            params[f"x{i+40}"] = 1.0
        self.optimizer.probe(params=params, lazy=True)

    def search(self, n_iter=1000):
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        return self.get_solution()

    def get_solution(self):

        skip_attn_layers = []
        for i in range(self.config.num_hidden_layers):
            if self.optimizer.max["params"][f"x{i}"] > 0.5:
                skip_attn_layers.append(i)

        skip_mlp_layers = []
        for i in range(
            self.config.num_hidden_layers, self.config.num_hidden_layers * 2
        ):
            if self.optimizer.max["params"][f"x{i}"] > 0.5:
                skip_mlp_layers.append(i - self.config.num_hidden_layers)

        return skip_attn_layers, skip_mlp_layers
