from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from .alerting_state import AlertingState

from .llm.schemas import Alert, EvidenceSpan

logger = logging.getLogger(__name__)