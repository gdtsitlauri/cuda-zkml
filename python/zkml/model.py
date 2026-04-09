"""
Model loading and conversion utilities.

Supports loading from:
- PyTorch state_dict (.pt, .pth)
- ONNX (.onnx)
- Raw float32 binary (.bin)

Converts model weights to quantized format suitable for ZK proving.
"""

import json
import numpy as np
import os
from typing import Any, Dict, List, Tuple, Optional


class Layer:
    """Represents a single neural network layer."""

    def __init__(self, layer_type: str, in_features: int, out_features: int, config: Optional[Dict[str, Any]] = None):
        self.layer_type = layer_type  # 'linear', 'relu', 'softmax'
        self.in_features = in_features
        self.out_features = out_features
        self.config: Dict[str, Any] = dict(config or {})
        self.weights: Optional[np.ndarray] = None  # [out, in] for linear
        self.bias: Optional[np.ndarray] = None      # [out] for linear
        self.q_weights: Optional[np.ndarray] = None
        self.q_bias: Optional[np.ndarray] = None
        self.k_weights: Optional[np.ndarray] = None
        self.k_bias: Optional[np.ndarray] = None
        self.v_weights: Optional[np.ndarray] = None
        self.v_bias: Optional[np.ndarray] = None
        self.o_weights: Optional[np.ndarray] = None
        self.o_bias: Optional[np.ndarray] = None

    def __repr__(self):
        return f"Layer({self.layer_type}, {self.in_features} -> {self.out_features})"


class Model:
    """Neural network model for ZK proving."""

    def __init__(self):
        self.layers: List[Layer] = []
        self.input_size: int = 0
        self.output_size: int = 0

    @staticmethod
    def _validate_layer_sequence(layers: List[Layer]) -> None:
        for idx, layer in enumerate(layers):
            if layer.layer_type not in {"linear", "relu", "softmax", "self_attention"}:
                raise ValueError(f"Unsupported layer type: {layer.layer_type}")
            if layer.layer_type == "softmax" and idx != len(layers) - 1:
                raise ValueError("Only final softmax layers are supported by the prover")

    @staticmethod
    def from_architecture_spec(spec: Dict[str, Any]) -> "Model":
        if "layers" not in spec or not isinstance(spec["layers"], list):
            raise ValueError("Architecture spec must contain a 'layers' list")

        model = Model()
        for layer_spec in spec["layers"]:
            layer_type = str(layer_spec["type"]).lower()
            in_features = int(layer_spec["in_features"])
            out_features = int(layer_spec["out_features"])
            config = {}
            if layer_type == "self_attention":
                config["seq_len"] = int(layer_spec["seq_len"])
                config["hidden_size"] = int(layer_spec["hidden_size"])
                config["num_heads"] = int(layer_spec.get("num_heads", 1))
            layer = Layer(layer_type, in_features, out_features, config=config)

            if layer_type == "linear":
                layer.weights = np.zeros((out_features, in_features), dtype=np.float32)
                layer.bias = np.zeros(out_features, dtype=np.float32)
            elif layer_type == "self_attention":
                hidden = int(config["hidden_size"])
                layer.q_weights = np.zeros((hidden, hidden), dtype=np.float32)
                layer.q_bias = np.zeros(hidden, dtype=np.float32)
                layer.k_weights = np.zeros((hidden, hidden), dtype=np.float32)
                layer.k_bias = np.zeros(hidden, dtype=np.float32)
                layer.v_weights = np.zeros((hidden, hidden), dtype=np.float32)
                layer.v_bias = np.zeros(hidden, dtype=np.float32)
                layer.o_weights = np.zeros((hidden, hidden), dtype=np.float32)
                layer.o_bias = np.zeros(hidden, dtype=np.float32)

            model.layers.append(layer)

        Model._validate_layer_sequence(model.layers)
        if model.layers:
            model.input_size = model.layers[0].in_features
            model.output_size = model.layers[-1].out_features
        return model

    @staticmethod
    def from_architecture_file(path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return Model.from_architecture_spec(spec)

    @staticmethod
    def from_onnx(path: str) -> "Model":
        """Load model from ONNX file."""
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            raise ImportError("pip install onnx to load ONNX models")

        model = Model()
        onnx_model = onnx.load(path)

        # Extract weights from initializers
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = numpy_helper.to_array(init)

        pending_linear: Optional[Layer] = None

        # Parse graph nodes
        for node in onnx_model.graph.node:
            if node.op_type in {"MatMul", "Gemm"}:
                # Find weight tensor
                w_name = node.input[1] if len(node.input) > 1 else None
                b_name = node.input[2] if len(node.input) > 2 else None

                if w_name and w_name in weights:
                    W = weights[w_name]
                    if W.ndim != 2:
                        raise ValueError(f"Unsupported weight rank for {node.name or node.op_type}: {W.shape}")
                    if node.op_type == "MatMul":
                        if W.shape[0] == 0 or W.shape[1] == 0:
                            raise ValueError(f"Invalid MatMul weight shape: {W.shape}")
                        if W.shape[0] >= W.shape[1]:
                            out_f, in_f = W.shape
                        else:
                            in_f, out_f = W.shape
                            W = W.T
                    else:
                        out_f, in_f = W.shape
                    layer = Layer("linear", in_f, out_f)
                    layer.weights = W.astype(np.float32)
                    if b_name and b_name in weights:
                        layer.bias = weights[b_name].astype(np.float32)
                    else:
                        layer.bias = np.zeros(out_f, dtype=np.float32)
                    model.layers.append(layer)
                    pending_linear = layer

            elif node.op_type == "Add":
                if pending_linear is not None and len(node.input) > 1:
                    bias_name = node.input[1]
                    if bias_name in weights:
                        bias = weights[bias_name].astype(np.float32).reshape(-1)
                        if pending_linear.bias is not None and bias.shape[0] == pending_linear.out_features:
                            pending_linear.bias = bias
                pending_linear = None

            elif node.op_type == "Relu":
                if model.layers:
                    prev_out = model.layers[-1].out_features
                    model.layers.append(Layer("relu", prev_out, prev_out))
                pending_linear = None

            elif node.op_type == "Softmax":
                if model.layers:
                    prev_out = model.layers[-1].out_features
                    model.layers.append(Layer("softmax", prev_out, prev_out))
                pending_linear = None

            elif node.op_type in {"Flatten", "Reshape", "Transpose", "Identity"}:
                continue
            else:
                raise ValueError(f"Unsupported ONNX op for current prover: {node.op_type}")

        Model._validate_layer_sequence(model.layers)
        if model.layers:
            model.input_size = model.layers[0].in_features
            model.output_size = model.layers[-1].out_features

        return model

    @staticmethod
    def from_pytorch(path: str, architecture: List[Tuple[str, int, int]]) -> "Model":
        """
        Load model from PyTorch state_dict.

        architecture: list of (layer_type, in_features, out_features)
        Example: [('linear', 784, 128), ('relu', 128, 128), ('linear', 128, 10)]
        """
        try:
            import torch
        except ImportError:
            raise ImportError("pip install torch to load PyTorch models")

        model = Model()
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        linear_idx = 0
        linear_keys = []
        for key in state_dict:
            if "weight" in key:
                linear_keys.append(key.replace(".weight", ""))

        for layer_type, in_f, out_f in architecture:
            layer = Layer(layer_type, in_f, out_f)

            if layer_type == "linear" and linear_idx < len(linear_keys):
                prefix = linear_keys[linear_idx]
                w_key = f"{prefix}.weight"
                b_key = f"{prefix}.bias"
                if w_key in state_dict:
                    layer.weights = state_dict[w_key].numpy().astype(np.float32)
                if b_key in state_dict:
                    layer.bias = state_dict[b_key].numpy().astype(np.float32)
                else:
                    layer.bias = np.zeros(out_f, dtype=np.float32)
                linear_idx += 1

            model.layers.append(layer)

        Model._validate_layer_sequence(model.layers)
        if model.layers:
            model.input_size = model.layers[0].in_features
            model.output_size = model.layers[-1].out_features

        return model

    @staticmethod
    def from_binary(path: str, architecture) -> "Model":
        """Load model weights from flat float32 binary."""
        if isinstance(architecture, dict):
            arch_spec = architecture
        else:
            layers = []
            for entry in architecture:
                if isinstance(entry, dict):
                    layers.append(entry)
                else:
                    layer_type, in_f, out_f = entry
                    layers.append(
                        {
                            "type": layer_type,
                            "in_features": in_f,
                            "out_features": out_f,
                        }
                    )
            arch_spec = {"layers": layers}

        model = Model.from_architecture_spec(arch_spec)

        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.float32)

        offset = 0
        for layer in model.layers:
            if layer.layer_type == "linear":
                in_f = layer.in_features
                out_f = layer.out_features
                n_weights = out_f * in_f
                n_bias = out_f
                layer.weights = data[offset:offset + n_weights].reshape(out_f, in_f)
                offset += n_weights
                layer.bias = data[offset:offset + n_bias]
                offset += n_bias
            elif layer.layer_type == "self_attention":
                hidden = int(layer.config["hidden_size"])
                n_weights = hidden * hidden
                n_bias = hidden

                def read_matrix() -> np.ndarray:
                    nonlocal offset
                    block = data[offset:offset + n_weights].reshape(hidden, hidden)
                    offset += n_weights
                    return block

                def read_bias() -> np.ndarray:
                    nonlocal offset
                    block = data[offset:offset + n_bias]
                    offset += n_bias
                    return block

                layer.q_weights = read_matrix()
                layer.q_bias = read_bias()
                layer.k_weights = read_matrix()
                layer.k_bias = read_bias()
                layer.v_weights = read_matrix()
                layer.v_bias = read_bias()
                layer.o_weights = read_matrix()
                layer.o_bias = read_bias()
        return model

    def to_binary(self, path: str):
        """Save model weights as flat float32 binary."""
        data = []
        for layer in self.layers:
            if layer.layer_type == "linear":
                data.append(layer.weights.flatten())
                data.append(layer.bias.flatten())
            elif layer.layer_type == "self_attention":
                data.extend(
                    [
                        layer.q_weights.flatten(),
                        layer.q_bias.flatten(),
                        layer.k_weights.flatten(),
                        layer.k_bias.flatten(),
                        layer.v_weights.flatten(),
                        layer.v_bias.flatten(),
                        layer.o_weights.flatten(),
                        layer.o_bias.flatten(),
                    ]
                )

        all_data = np.concatenate(data).astype(np.float32)
        all_data.tofile(path)

    def architecture_spec(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layers": [
                {
                    "type": layer.layer_type,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    **(
                        {
                            "seq_len": int(layer.config["seq_len"]),
                            "hidden_size": int(layer.config["hidden_size"]),
                            "num_heads": int(layer.config.get("num_heads", 1)),
                        }
                        if layer.layer_type == "self_attention"
                        else {}
                    ),
                }
                for layer in self.layers
            ],
        }

    def save_architecture(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.architecture_spec(), f, indent=2)
            f.write("\n")

    def summary(self):
        """Print model summary."""
        print(f"Model: {self.input_size} -> {self.output_size}")
        print(f"Layers: {len(self.layers)}")
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = 0
            if layer.weights is not None:
                params += layer.weights.size
            if layer.bias is not None:
                params += layer.bias.size
            if layer.layer_type == "self_attention":
                params += int(layer.q_weights.size + layer.q_bias.size)
                params += int(layer.k_weights.size + layer.k_bias.size)
                params += int(layer.v_weights.size + layer.v_bias.size)
                params += int(layer.o_weights.size + layer.o_bias.size)
            total_params += params
            print(f"  [{i}] {layer.layer_type}: "
                  f"{layer.in_features} -> {layer.out_features} "
                  f"({params} params)")
        print(f"Total parameters: {total_params}")

    def float_inference(self, x: np.ndarray) -> np.ndarray:
        """Run float32 inference for comparison."""
        for layer in self.layers:
            if layer.layer_type == "linear":
                x = x @ layer.weights.T + layer.bias
            elif layer.layer_type == "relu":
                x = np.maximum(0, x)
            elif layer.layer_type == "softmax":
                e = np.exp(x - np.max(x))
                x = e / e.sum()
            elif layer.layer_type == "self_attention":
                x = self._self_attention_forward(layer, x)
        return x

    @staticmethod
    def _self_attention_forward(layer: Layer, x: np.ndarray) -> np.ndarray:
        seq_len = int(layer.config["seq_len"])
        hidden = int(layer.config["hidden_size"])
        num_heads = int(layer.config.get("num_heads", 1))
        if num_heads <= 0 or hidden % num_heads != 0:
            raise ValueError("self_attention num_heads must divide hidden_size")
        head_dim = hidden // num_heads
        tokens = x.reshape(seq_len, hidden)

        q = tokens @ layer.q_weights.T + layer.q_bias
        k = tokens @ layer.k_weights.T + layer.k_bias
        v = tokens @ layer.v_weights.T + layer.v_bias
        context = np.zeros_like(tokens)
        for head in range(num_heads):
            start = head * head_dim
            end = start + head_dim
            qh = q[:, start:end]
            kh = k[:, start:end]
            vh = v[:, start:end]
            scores = qh @ kh.T
            exp_scores = 1.0 + scores + scores * scores + scores * scores * scores
            exp_scores = np.maximum(exp_scores, 1e-6)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            context[:, start:end] = probs @ vh
        out = context @ layer.o_weights.T + layer.o_bias
        out = out + tokens
        return out.reshape(-1).astype(np.float32)

    @staticmethod
    def create_mnist_mlp() -> "Model":
        """Create a random MNIST MLP for testing."""
        model = Model.create_mlp(784, [128], 10, seed=42)
        return model

    @staticmethod
    def create_cifar_mlp() -> "Model":
        """Create a random CIFAR-10 MLP on flattened RGB inputs."""
        return Model.create_mlp(3072, [256, 64], 10, seed=123)

    @staticmethod
    def create_tiny_transformer() -> "Model":
        """Create a tiny single-head transformer-like classifier for testing."""
        model = Model()
        model.input_size = 32  # 4 tokens x 8 hidden dim
        model.output_size = 4
        rng = np.random.RandomState(321)

        attn = Layer("self_attention", 32, 32, config={"seq_len": 4, "hidden_size": 8, "num_heads": 1})
        attn.q_weights = rng.randn(8, 8).astype(np.float32) * 0.05
        attn.q_bias = np.zeros(8, dtype=np.float32)
        attn.k_weights = rng.randn(8, 8).astype(np.float32) * 0.05
        attn.k_bias = np.zeros(8, dtype=np.float32)
        attn.v_weights = rng.randn(8, 8).astype(np.float32) * 0.05
        attn.v_bias = np.zeros(8, dtype=np.float32)
        attn.o_weights = rng.randn(8, 8).astype(np.float32) * 0.05
        attn.o_bias = np.zeros(8, dtype=np.float32)
        model.layers.append(attn)
        model.layers.append(Layer("relu", 32, 32))

        out = Layer("linear", 32, 4)
        out.weights = rng.randn(4, 32).astype(np.float32) * 0.05
        out.bias = np.zeros(4, dtype=np.float32)
        model.layers.append(out)
        model.layers.append(Layer("softmax", 4, 4))
        return model

    @staticmethod
    def create_multihead_transformer(num_heads: int = 2, num_blocks: int = 2) -> "Model":
        """Create a small stacked multi-head transformer-like classifier for testing."""
        if num_heads <= 0 or 8 % num_heads != 0:
            raise ValueError("num_heads must divide hidden size 8")

        model = Model()
        model.input_size = 32
        model.output_size = 4
        rng = np.random.RandomState(777)

        for _ in range(num_blocks):
            attn = Layer(
                "self_attention",
                32,
                32,
                config={"seq_len": 4, "hidden_size": 8, "num_heads": num_heads},
            )
            attn.q_weights = rng.randn(8, 8).astype(np.float32) * 0.05
            attn.q_bias = np.zeros(8, dtype=np.float32)
            attn.k_weights = rng.randn(8, 8).astype(np.float32) * 0.05
            attn.k_bias = np.zeros(8, dtype=np.float32)
            attn.v_weights = rng.randn(8, 8).astype(np.float32) * 0.05
            attn.v_bias = np.zeros(8, dtype=np.float32)
            attn.o_weights = rng.randn(8, 8).astype(np.float32) * 0.05
            attn.o_bias = np.zeros(8, dtype=np.float32)
            model.layers.append(attn)
            model.layers.append(Layer("relu", 32, 32))

        out = Layer("linear", 32, 4)
        out.weights = rng.randn(4, 32).astype(np.float32) * 0.05
        out.bias = np.zeros(4, dtype=np.float32)
        model.layers.append(out)
        model.layers.append(Layer("softmax", 4, 4))
        return model

    @staticmethod
    def create_mlp(
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        seed: int = 42,
        final_softmax: bool = True,
    ) -> "Model":
        model = Model()
        model.input_size = input_size
        model.output_size = output_size
        prev_size = input_size
        rng = np.random.RandomState(seed)

        for hidden_size in hidden_sizes:
            linear = Layer("linear", prev_size, hidden_size)
            linear.weights = rng.randn(hidden_size, prev_size).astype(np.float32) * 0.01
            linear.bias = np.zeros(hidden_size, dtype=np.float32)
            model.layers.append(linear)
            model.layers.append(Layer("relu", hidden_size, hidden_size))
            prev_size = hidden_size

        out = Layer("linear", prev_size, output_size)
        out.weights = rng.randn(output_size, prev_size).astype(np.float32) * 0.01
        out.bias = np.zeros(output_size, dtype=np.float32)
        model.layers.append(out)

        if final_softmax:
            model.layers.append(Layer("softmax", output_size, output_size))

        return model
