"""Shared pytest configuration for analyses tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "notebook: execute a Jupyter notebook top-to-bottom (slow, needs real data)",
    )
