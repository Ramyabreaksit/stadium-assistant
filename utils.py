import os
import logging
import importlib.metadata
from typing import Optional, List


def setup_logging(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized logger instance with consistent formatting.

    Args:
        logger_name (str): Name of the logger module (`__name__`).
        level (int): Logging severity level (default: `logging.INFO`).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logging(__name__)


def verify_requirements_versions(requirements_path: Optional[str] = None) -> bool:
    """Verify that installed packages meet the minimum or exact versions specified in `requirements.txt`.

    Args:
        requirements_path (Optional[str]): Path to `requirements.txt`. If `None`, locates file next to project root.

    Returns:
        bool: `True` if all checked requirements are satisfied, `False` if warnings or errors occurred.
    """
    if requirements_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        requirements_path = os.path.join(base_dir, "requirements.txt")

    if not os.path.exists(requirements_path):
        logger.warning("Requirements file not found at %s. Skipping version pin check.", requirements_path)
        return False

    all_satisfied: bool = True
    try:
        with open(requirements_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        for line in lines:
            pkg_name = line
            op = None
            target_ver = None
            
            if ">=" in line:
                pkg_name, target_ver = line.split(">=", 1)
                op = ">="
            elif "==" in line:
                pkg_name, target_ver = line.split("==", 1)
                op = "=="
            
            pkg_name = pkg_name.strip()
            if target_ver:
                target_ver = target_ver.strip()

            try:
                installed_ver = importlib.metadata.version(pkg_name)
                
                if target_ver and op == ">=":
                    installed_parts = [int(p) for p in installed_ver.split(".") if p.isdigit()]
                    target_parts = [int(p) for p in target_ver.split(".") if p.isdigit()]
                    if installed_parts < target_parts:
                        logger.warning("Package '%s' version %s is lower than required %s.", pkg_name, installed_ver, target_ver)
                        all_satisfied = False
                    else:
                        logger.debug("Verified requirement: %s (%s >= %s)", pkg_name, installed_ver, target_ver)
                elif target_ver and op == "==":
                    if installed_ver != target_ver:
                        logger.warning("Package '%s' version %s does not match exact requirement %s.", pkg_name, installed_ver, target_ver)
                        all_satisfied = False
                    else:
                        logger.debug("Verified exact requirement: %s (%s == %s)", pkg_name, installed_ver, target_ver)
                else:
                    logger.debug("Verified presence of package: %s (%s)", pkg_name, installed_ver)
            except importlib.metadata.PackageNotFoundError:
                logger.error("Required package '%s' is not installed in the active environment.", pkg_name)
                all_satisfied = False

    except Exception as e:
        logger.error("Error during requirements verification: %s", e)
        return False

    if all_satisfied:
        logger.info("All package dependencies in requirements.txt verified successfully.")
    return all_satisfied
