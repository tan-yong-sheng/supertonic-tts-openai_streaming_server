import pytest

from app import create_app
from app.config import Config


@pytest.fixture()
def client():
    Config.AUTHENTICATION_ALLOWED_TOKENS = None
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client
