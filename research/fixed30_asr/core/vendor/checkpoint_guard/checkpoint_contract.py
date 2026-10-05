"""Version 1 strict checkpoint contracts, using only the standard library.

This is newly authored preparation code, not recovered historical source.
Tensor inspection is delegated to a separately audited, trusted runtime callback.
No tensor runtime is imported and no weight application occurs in this module.
"""
from collections import OrderedDict
import hashlib
import json
import unicodedata

SCHEMA = "sensevoice-exact-checkpoint-contract-v1"
DTYPES = frozenset(("torch.float32", "torch.float64", "torch.int64", "torch.int32",
                    "torch.int16", "torch.int8", "torch.uint8", "torch.bool"))
MAX_KEYS = 100000
MAX_DIMS = 16
SENSEVOICE_METADATA_RECIPE = "sensevoice-module-version-1-metadata-v1"
SENSEVOICE_SCHEMA_SHA256 = "7a5112921dc839499bc4b6b88504ecd75dd6332ebd4f6131ae5a78057ae28ae0"
SENSEVOICE_ARCH_METADATA_SHA256 = "5471f36f6c3e5b6295d918347d60b0c9471cf03496f150c5072948c3cbfa14de"
SENSEVOICE_CHECKPOINT_METADATA_SHA256 = "317d50b5306ee6f5168b453f590f1767066a2f3b80acd0451b9d34de3fbc5147"


def sensevoice_architecture_metadata():
    """Independent module topology from the reviewed 50+20 SANM constructors.

    Runtime comparison qualified torch 2.12.1 and funasr 1.4.16. Every module
    uses Module._load_from_state_dict without load/state hooks and version 1.
    Runtime/version/source qualification remains the trusted caller's duty.
    This derivation does not read checkpoint contents or infer tensor keys.
    """
    names = ["", "criterion_att", "criterion_att.criterion", "ctc", "ctc.ctc_lo",
             "ctc.ctc_loss", "embed", "encoder", "encoder.after_norm", "encoder.embed",
             "encoder.encoders", "encoder.encoders0", "encoder.tp_encoders",
             "encoder.tp_norm", "specaug", "specaug.freq_mask", "specaug.time_mask"]
    layers = ["encoder.encoders0.0"] + [f"encoder.encoders.{i}" for i in range(49)]
    layers += [f"encoder.tp_encoders.{i}" for i in range(20)]
    suffixes = ("", ".dropout", ".feed_forward", ".feed_forward.activation",
                ".feed_forward.dropout", ".feed_forward.w_1", ".feed_forward.w_2",
                ".norm1", ".norm2", ".self_attn", ".self_attn.dropout",
                ".self_attn.fsmn_block", ".self_attn.linear_out",
                ".self_attn.linear_q_k_v", ".self_attn.pad_fn")
    names += [name + suffix for name in layers for suffix in suffixes]
    return {name: {"version": 1} for name in names}


def snapshot_metadata(state, label):
    """Accept only PyTorch's exact version-1 primitive metadata structure.

    Keep the original dict/OrderedDict choice and entry order in the detached
    copy supplied to the loader. Arbitrary attributes, subclasses, additional
    fields (including assign_to_params_buffers), and coercions are forbidden.
    """
    need(type(state) is OrderedDict and set(state.__dict__) == {"_metadata"},
         label + " must have only the reviewed _metadata attribute")
    value = state.__dict__["_metadata"]
    need(type(value) in (dict, OrderedDict), label + " metadata must be a direct mapping")
    need(type(value) is not OrderedDict or not value.__dict__,
         label + " metadata mapping has unexpected attributes")
    need(0 < len(value) <= MAX_KEYS, label + " metadata must be bounded and nonempty")
    copied = type(value)()
    for key, row in value.items():
        need(type(key) is str and len(key) <= 1024 and
             (key == "" or all(key.split("."))) and
             not any(c.isspace() or unicodedata.category(c).startswith("C") for c in key),
             label + " metadata has an invalid module name")
        need(type(row) is dict and set(row) == {"version"} and
             type(row["version"]) is int and row["version"] == 1,
             label + " metadata must contain only exact integer version 1")
        copied[key] = {"version": row["version"]}
    return copied


def prepare_metadata(checkpoint, model_state, expected_schema_sha256, recipe):
    need(type(recipe) is str and recipe == SENSEVOICE_METADATA_RECIPE,
         "Unknown checkpoint metadata recipe")
    need(expected_schema_sha256 == SENSEVOICE_SCHEMA_SHA256,
         "Metadata recipe is restricted to the independently approved SenseVoice schema")
    architecture = snapshot_metadata(model_state, "Architecture")
    checkpoint_metadata = snapshot_metadata(checkpoint, "Checkpoint")
    expected_architecture = sensevoice_architecture_metadata()
    need(digest(expected_architecture) == SENSEVOICE_ARCH_METADATA_SHA256,
         "Independent architecture metadata recipe changed")
    need(architecture == expected_architecture,
         "Architecture metadata differs from reviewed module topology/version")
    # The released checkpoint retains these two module-version records. They
    # have no tensors and no corresponding module in the qualified runtime.
    # Preserve and disclose them; do not drop them or generalize to other keys.
    unused = {"frontend": {"version": 1}, "encoder.rwkv_encoders": {"version": 1}}
    expected_checkpoint = dict(expected_architecture, **unused)
    need(checkpoint_metadata == expected_checkpoint and
         digest(checkpoint_metadata) == SENSEVOICE_CHECKPOINT_METADATA_SHA256,
         "Checkpoint metadata differs from exact reviewed released metadata")
    need(not any(key == name or key.startswith(name + ".")
                 for name in unused for key in checkpoint),
         "Checkpoint-only metadata module unexpectedly owns a tensor")
    return checkpoint_metadata, {
        "recipe": recipe,
        "qualified_runtime_versions": {"torch": "2.12.1", "funasr": "1.4.16"},
        "checkpoint": {key: dict(row) for key, row in checkpoint_metadata.items()},
        "checkpoint_sha256": digest(checkpoint_metadata),
        "architecture": {key: dict(row) for key, row in architecture.items()},
        "architecture_sha256": digest(architecture),
        "checkpoint_only_module_records": unused,
        "all_checkpoint_metadata_preserved": True,
        "module_version_migration_allowed": False,
    }


def verify_metadata(state, contract, which):
    metadata_contract = contract.get("metadata")
    if metadata_contract is not None:
        need(snapshot_metadata(state, which) == metadata_contract[which],
             which + " metadata changed after prevalidation")
        need(digest(metadata_contract[which]) == metadata_contract[which + "_sha256"],
             which + " metadata contract digest changed")


class CheckpointContractError(RuntimeError):
    def __init__(self, message, *, stage="prevalidation", mutation_may_have_occurred=False):
        super().__init__(message)
        self.stage = stage
        self.mutation_may_have_occurred = mutation_may_have_occurred


def need(value, message):
    if not value:
        raise CheckpointContractError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_mapping(value, label):
    # Deliberately no wrapper-unwrapping, custom mapping or prefix heuristics.
    need(type(value) in (dict, OrderedDict), label + " must be direct dict or OrderedDict")
    need(0 < len(value) <= MAX_KEYS, label + " must have bounded nonempty state")
    for key in value:
        need(type(key) is str and 0 < len(key) <= 1024 and
             not any(c.isspace() or unicodedata.category(c).startswith("C") for c in key),
             label + " has invalid state key")
    return value


def validate_description(value, include_content):
    wanted = {"shape", "dtype", "sha256"} if include_content else {"shape", "dtype"}
    need(type(value) is dict and set(value) == wanted, "Tensor descriptor fields differ")
    shape, dtype = value["shape"], value["dtype"]
    need(type(shape) is list and len(shape) <= MAX_DIMS and
         all(type(n) is int and 0 <= n <= (1 << 40) for n in shape), "Invalid tensor shape")
    need(type(dtype) is str and dtype in DTYPES, "Unsupported tensor dtype recipe")
    row = {"shape": list(shape), "dtype": dtype}
    if include_content:
        need(valid_sha(value["sha256"]), "Invalid tensor content SHA256")
        row["sha256"] = value["sha256"]
    return row


def validate_schema(schema, approved_sha256):
    validate_mapping(schema, "Expected schema")
    need(valid_sha(approved_sha256), "Independently approved schema SHA256 is required")
    copied = {key: validate_description(value, False) for key, value in schema.items()}
    need(digest(copied) == approved_sha256, "Expected schema SHA256 mismatch")
    return copied


def describe_state(state, describe_tensor, include_content):
    validate_mapping(state, "State")
    need(callable(describe_tensor), "A trusted tensor descriptor is required")
    result = {}
    for key in sorted(state):
        result[key] = validate_description(describe_tensor(state[key], include_content), include_content)
    return result


def schema_of(descriptions):
    return {key: {"shape": list(row["shape"]), "dtype": row["dtype"]}
            for key, row in descriptions.items()}


def prepare_contract(checkpoint, model_state, describe_tensor, *, expected_schema,
                     expected_schema_sha256, prefix_mappings=(("", ""),),
                     metadata_recipe=None):
    """Validate every key and descriptor without applying model weights.

    expected_schema must originate from independently audited architecture/config,
    never be derived from this checkpoint merely to make it pass.
    """
    need(type(prefix_mappings) is tuple and len(prefix_mappings) == 1 and
         type(prefix_mappings[0]) is tuple and len(prefix_mappings[0]) == 2 and
         all(type(v) is str for v in prefix_mappings[0]) and
         prefix_mappings == (("", ""),),
         "Only exact identity key mapping is supported")
    schema = validate_schema(expected_schema, expected_schema_sha256)
    validate_mapping(checkpoint, "Checkpoint")
    if metadata_recipe is None:
        need(type(checkpoint) is not OrderedDict or not checkpoint.__dict__,
             "Checkpoint OrderedDict attributes/metadata require a separately reviewed recipe")
    validate_mapping(model_state, "Architecture state")
    need(set(checkpoint) == set(schema), "Checkpoint key coverage differs from expected schema")
    need(set(model_state) == set(schema), "Architecture key coverage differs from expected schema")
    # Snapshot the container before callbacks. Tensor values still require exclusive
    # ownership; this is not a sandbox against hostile callbacks or concurrent writes.
    frozen_checkpoint = OrderedDict(checkpoint.items())
    metadata_contract = None
    if metadata_recipe is not None:
        frozen_checkpoint._metadata, metadata_contract = prepare_metadata(
            checkpoint, model_state, expected_schema_sha256, metadata_recipe)
    checkpoint_description = describe_state(frozen_checkpoint, describe_tensor, True)
    architecture_description = describe_state(model_state, describe_tensor, False)
    need(schema_of(checkpoint_description) == schema, "Checkpoint shape/dtype differs from expected schema")
    need(architecture_description == schema, "Architecture shape/dtype differs from expected schema")
    contract = {
        "schema": SCHEMA, "expected_schema_sha256": expected_schema_sha256,
        "expected_schema": schema, "checkpoint_state": checkpoint_description,
        "checkpoint_state_sha256": digest(checkpoint_description),
        "identity_key_mapping_only": True,
    }
    if metadata_contract is not None:
        contract["metadata"] = metadata_contract
    return frozen_checkpoint, contract
