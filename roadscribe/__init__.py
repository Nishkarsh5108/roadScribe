"""RoadScribe: grounded trip narratives from vehicular IMU + GNSS and road topology.

Layer 1 (``roadscribe.layer1``) turns raw inertial and GNSS samples into
vehicle-frame ride measures and events; Layer 2 (``roadscribe.layer2``)
segments the trip into homogeneous road narrative units (HRNUs); Layer 3
(``roadscribe.layer3``) compiles the claim ledger and the token-budgeted
capsule. See ``roadscribe_research/METHODOLOGY.md``.
"""
from .config import RoadScribeConfig, config_hash
from .schemas import HRNU, Capsule, Claim, Event, Fact, Narrative, RawTrip, VehicleFrameTrip

__version__ = "0.1.0"

__all__ = [
    "RoadScribeConfig",
    "config_hash",
    "RawTrip",
    "VehicleFrameTrip",
    "Event",
    "HRNU",
    "Fact",
    "Capsule",
    "Claim",
    "Narrative",
    "__version__",
]
