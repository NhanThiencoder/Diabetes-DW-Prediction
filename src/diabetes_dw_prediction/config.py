from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_interim_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def reports_figures_dir(self) -> Path:
        return self.project_root / "reports" / "figures"


def get_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return ProjectPaths(project_root=root)
