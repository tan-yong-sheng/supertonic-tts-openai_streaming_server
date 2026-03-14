from app.config import Config


def test_models_endpoint(client):
    resp = client.get('/v1/models')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['object'] == 'list'
    ids = [item['id'] for item in data['data']]
    assert ids == Config.ALLOWED_MODELS
