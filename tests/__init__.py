# -*- coding: utf-8 -*-
"""
Testing framework for license plate recognition system
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

__all__ = ['TestSuite', 'UnitTestSuite', 'IntegrationTestSuite', 'PerformanceTestSuite']

from .test_suite import TestSuite
from .unit.test_core import UnitTestSuite
from .integration.test_pipeline import IntegrationTestSuite
from .performance.test_benchmarks import PerformanceTestSuite
