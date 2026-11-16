import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CellSettings:
    feed_rate_milling: float
    feed_rate_gripping_and_removing_approaching: float
    feed_rate_gripping_and_removing_leaving: float
    feed_rate_screwing_approaching: float
    feed_rate_screwing_leaving: float
    toolchange_time_dropoff_to_storage: float
    toolchange_time_assembly_to_storage: float
    toolchange_time_at_toolstorage: float
    toolchange_time_storage_to_assembly: float
    toolchange_time_start_to_toolstorage: float
    traveltime_assembly_to_dropoff: float
    traveltime_dropoff_to_assembly: float
    screwing_start_offset: float
    screwing_pullout_factor: float
    screwing_minimal_pullout_value: float
    gripping_start_offset: float
    gripping_pullout_factor: float
    gripping_minimal_pullout_value: float
    milling_penalty: float

    @classmethod
    def from_json(cls, path: str | Path) -> "CellSettings":
        data = json.loads(Path(path).read_text())
        return cls(**data)
