"""Test configuration for :mod:`torchcor.mechanics`.

pytest installs its own warning filters around every test, which overrides the
one the package sets on import, so the same PyTorch beta-status notice has to
be silenced here as well.  See ``torchcor/mechanics/__init__.py``.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Sparse CSR tensor support is in beta state:UserWarning")
