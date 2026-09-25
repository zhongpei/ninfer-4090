"""Replace an existing NInfer text tower with a PrismML GGUF text tower.

The input artifact remains the source of truth for the model directory,
resources, vision tower, MTP, DFlash2, proposal configuration, and every
non-text tensor.  Only logical text/proposal tensors and their shared
Hadamard-sign auxiliaries are regenerated from the selected GGUF.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorObject, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.__main__ import SourceInputs
from tools.convert.qwen3_5 import build_model
from tools.convert.proposal import add_official_proposal
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.ternary import (
    bonsai2_27b_ternary,
    mixed_projection_format,
    validate as validate_gguf,
)


DEFAULT_GGUF = Path(
    "/opt/llama.cpp-Ternary-Bonsai-2-27B/models/"
    "Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf"
)
DEFAULT_ARTIFACT = Path("/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer")
TEXT_PREFIXES = ("text/", "proposal/")


def _binding_object_ids(binding: dict) -> tuple[str, ...]:
    if "object" in binding:
        return (binding["object"],)
    return tuple(part["object"] for part in binding["parts"])


def _object_spec(obj: ResourceObject | TensorObject):
    if isinstance(obj, ResourceObject):
        return ResourceSpec(obj.id, obj.bytes, obj.encoding)
    return TensorSpec(obj.id, obj.shape, obj.format, obj.layout)


def _replacement_spec(old: TensorObject, job) -> TensorSpec:
    """Return the output spec, allowing only the registered mixed transition."""
    expected = job.spec
    if old.shape != expected.shape or old.layout != expected.layout:
        raise ValueError(
            f"{old.id}: existing tensor {(old.shape, old.format, old.layout)} "
            f"differs from new tensor {(expected.shape, expected.format, expected.layout)}"
        )
    if old.format != expected.format:
        allowed = (
            old.format == "t2_g128_fp16"
            and expected.format == "q5_g64_fp16"
            and len(job.parameters) == 1
            and mixed_projection_format(job.parameters[0]) == expected.format
        )
        if not allowed:
            raise ValueError(
                f"{old.id}: unregistered format transition "
                f"{old.format} -> {expected.format}"
            )
    return TensorSpec(old.id, expected.shape, expected.format, expected.layout)


def _text_parameter_names(bindings: dict) -> set[str]:
    return {name for name in bindings if name.startswith(TEXT_PREFIXES)}


def _materialize_base(artifact: Artifact, root: Path) -> None:
    """Create the config/resource-only source expected by the converter."""
    root.mkdir(parents=True, exist_ok=True)
    config = artifact.directory.components["text"]["config"]
    (root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    resources = artifact.directory.components["text"].get("resources", {})
    for role, object_id in resources.items():
        (root / role).write_bytes(artifact.read_object(object_id))


def _prepare_recipe(
    artifact: Artifact,
    gguf_path: Path,
    *,
    device: str,
    rows_per_chunk: int,
    include_proposal: bool,
    stack: ExitStack,
):
    """Build the replacement jobs while all lazy source handles stay open."""
    with tempfile.TemporaryDirectory(prefix="ninfer-huihui-") as temporary:
        base_root = Path(temporary) / "base"
        _materialize_base(artifact, base_root)
        base = stack.enter_context(SafetensorsSource(base_root))
        sources = SourceInputs(base, {"ternary": gguf_path}, stack)
        gguf = sources["ternary"]
        validate_gguf(gguf)
        model = build_model(base, components=("text",))
        recipe = Recipe(model)
        bonsai2_27b_ternary(model, recipe, sources)
        if include_proposal:
            rows = artifact.directory.components["text"]["proposal"]["rows"]
            add_official_proposal(recipe, rows=rows)
        prepared = recipe.prepare(device=device, rows_per_chunk=rows_per_chunk)

        # All selected text sources are now GGUF-backed and model resources were
        # materialised into Model.resources, so the temporary config directory can
        # be removed after preparation.
        return prepared


def _replacement_targets(artifact: Artifact, prepared) -> tuple[dict, dict, set[str]]:
    """Match newly prepared logical jobs to the old artifact's physical IDs."""
    expected_names = _text_parameter_names(artifact.directory.bindings)
    prepared_names = {
        parameter for job in prepared.weights for parameter in job.parameters
    }
    if prepared_names != expected_names:
        missing = sorted(expected_names - prepared_names)
        extra = sorted(prepared_names - expected_names)
        raise ValueError(
            "replacement parameter set differs from the existing artifact; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    jobs = {}
    used = {}
    for job in prepared.weights:
        target_ids = {
            object_id
            for parameter in job.parameters
            for object_id in _binding_object_ids(artifact.directory.bindings[parameter])
        }
        if len(target_ids) != 1:
            raise ValueError(
                f"{job.parameters}: existing bindings do not identify one physical object"
            )
        target_id = next(iter(target_ids))
        if target_id in used:
            raise ValueError(
                f"{target_id}: multiple replacement jobs map to the same object "
                f"({used[target_id]} and {job.parameters})"
            )
        old = artifact.by_id[target_id]
        if not isinstance(old, TensorObject):
            raise ValueError(f"{target_id}: replacement target is not a tensor")
        _replacement_spec(old, job)
        jobs[target_id] = job
        used[target_id] = job.parameters

    auxiliaries = {}
    for spec, data in prepared.auxiliaries:
        old = artifact.by_id.get(spec.id)
        if not isinstance(old, TensorObject):
            raise ValueError(f"{spec.id}: generated auxiliary is absent from the artifact")
        actual = (old.shape, old.format, old.layout)
        expected = (spec.shape, spec.format, spec.layout)
        if actual != expected:
            raise ValueError(
                f"{spec.id}: existing auxiliary {actual} differs from new auxiliary {expected}"
            )
        auxiliaries[spec.id] = data
    return jobs, auxiliaries, set(jobs) | set(auxiliaries)


def _copy_unchanged_objects(
    writer: ArtifactWriter, artifact: Artifact, replaced: set[str]
) -> None:
    for obj in artifact.objects:
        if obj.id not in replaced:
            writer.write_object(obj.id, artifact.iter_object(obj.id))


def _run(
    artifact_path: Path,
    gguf_path: Path,
    output: Path,
    *,
    device: str,
    rows_per_chunk: int,
    max_file_bytes: int,
    validate_only: bool,
) -> None:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"base artifact does not exist: {artifact_path}")
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF does not exist: {gguf_path}")
    if not validate_only and (output.exists() or Path(str(output) + ".conversion.json").exists()):
        raise FileExistsError(f"conversion output already exists: {output}")

    with Artifact(artifact_path) as artifact, ExitStack() as stack:
        include_proposal = "proposal" in artifact.directory.components["text"]
        prepared = _prepare_recipe(
            artifact,
            gguf_path,
            device=device,
            rows_per_chunk=rows_per_chunk,
            include_proposal=include_proposal,
            stack=stack,
        )
        jobs, auxiliaries, replaced = _replacement_targets(artifact, prepared)
        if validate_only:
            print(
                f"validated {gguf_path}: {len(jobs)} tensor jobs, "
                f"{len(auxiliaries)} shared auxiliaries, "
                f"preserving {len(artifact.objects) - len(replaced)} objects"
            )
            return

        specs = [
            _replacement_spec(obj, jobs[obj.id])
            if obj.id in jobs
            else _object_spec(obj)
            for obj in artifact.objects
        ]
        provenance = deepcopy(artifact.directory.provenance)
        provenance["text_replacement"] = {
            "converter": "tools/convert/huihui_bonsai/convert.py",
            "base_artifact": str(artifact_path),
            "ternary_gguf": str(gguf_path),
            "preserved_components": [
                name
                for name in artifact.directory.components
                if name not in ("text",)
            ],
        }
        with ArtifactWriter(
            output,
            specs,
            components=artifact.directory.components,
            bindings=artifact.directory.bindings,
            uses=artifact.directory.uses,
            metadata=artifact.directory.metadata,
            provenance=provenance,
            max_file_bytes=max_file_bytes,
        ) as writer:
            _copy_unchanged_objects(writer, artifact, replaced)
            for index, (target_id, job) in enumerate(jobs.items(), start=1):
                print(
                    f"[{index}/{len(jobs)}] {target_id}: "
                    f"{job.spec.format} {job.spec.shape}",
                    flush=True,
                )
                job.prepared.produce(TensorOutput(writer, target_id))
            for object_id, data in auxiliaries.items():
                writer.write_object(object_id, data)

        report = {
            "output": str(output),
            "base_artifact": str(artifact_path),
            "ternary_gguf": str(gguf_path),
            "replaced_tensor_objects": len(jobs),
            "replaced_auxiliary_objects": len(auxiliaries),
            "preserved_objects": len(artifact.objects) - len(replaced),
        }
        Path(str(output) + ".conversion.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"wrote {output}: replaced {len(jobs)} tensors and "
            f"{len(auxiliaries)} auxiliaries; preserved "
            f"{len(artifact.objects) - len(replaced)} objects"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--max-file-bytes", type=int, default=32_000_000_000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate GGUF, bindings, shapes, and resources without writing an artifact",
    )
    args = parser.parse_args(argv)
    if not args.validate_only and args.out is None:
        parser.error("--out is required unless --validate-only is used")
    _run(
        args.base_artifact,
        args.gguf,
        args.out,
        device=args.device,
        rows_per_chunk=args.rows_per_chunk,
        max_file_bytes=args.max_file_bytes,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
