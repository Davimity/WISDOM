"""Candidate discovery modes for reproducible and production dataset builds."""

from enum import Enum


class DiscoveryMode(str, Enum):
    """Select bounded fixtures or the official live RCSB search service."""

    FIXTURE = "fixture"
    LIVE    = "live"
