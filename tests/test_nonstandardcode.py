import importlib.util
import subprocess
import sys


def test_ispackage_installed():
    """
    verifies the package installation
    """
    package_name = "ml_package"

    assert (
        importlib.util.find_spec(package_name) is not None
    ), f"{package_name} is not installed"

    result = subprocess.run([sys.executable, "-m", "pip", "show", package_name])
    assert result.returncode == 0, f"Failed to find package {package_name}"
