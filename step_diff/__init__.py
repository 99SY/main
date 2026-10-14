"""B-Rep based object correspondence and design-change detection for STEP files."""
from .config import Config
from .pipeline import compare_objects, compare_step_files, extract_all

__all__ = ["Config", "compare_objects", "compare_step_files", "extract_all"]
__version__ = "0.1.0"
