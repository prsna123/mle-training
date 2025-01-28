import importlib.util


def test_ispackage_installed():
    """
    verifies the package installation
    """
    package_name = "ml_package"

    assert (
        importlib.util.find_spec(package_name) is not None
    ), f"{package_name} is not installed"
