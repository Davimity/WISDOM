"""The small metadata contract separating scientific tasks from the predictor."""

from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class TaskSpecification:
    """Name the global target and evaluation sidecar without task branches in Training."""

    task_name             : str = "dna_binding"
    global_target_key     : str = "dna_binding"
    local_annotation_asset: str = "dna_annotation"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "TaskSpecification":
        """Read explicit task metadata with a historical DNA-only fallback.

        Args:
            metadata: DatasetMember metadata containing optional ``task_specification``.

        Returns:
            Immutable target/annotation vocabulary; legacy members retain their DNA contract.

        Raises:
            TypeError: If authored task metadata is not a mapping.
            KeyError: If an explicit specification omits required fields.
        """
        value = metadata.get("task_specification")
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("task_specification must be a JSON object")
        return cls(str(value["task_name"]), str(value["global_target_key"]),
                   str(value["local_annotation_asset"]))
