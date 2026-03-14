import pytest

@pytest.mark.unit
def test_invalid_model_is_case_sensitive(client):
    resp = client.post(
        '/v1/audio/speech',
        json={
            'model': 'Supertonic-2',
            'input': 'Hello',
            'voice': 'M1',
            'response_format': 'mp3',
        },
    )
    assert resp.status_code == 401
    data = resp.get_json()
    err = data['error']
    assert err['param'] == 'model'
    assert 'allowed_models' in err['details']
    assert 'suggested_defaults' in err['details']


@pytest.mark.unit
def test_invalid_voice_returns_401(client):
    resp = client.post(
        '/v1/audio/speech',
        json={
            'model': 'supertonic-2',
            'input': 'Hello',
            'voice': 'NOPE',
            'response_format': 'mp3',
        },
    )
    assert resp.status_code == 401
    data = resp.get_json()
    err = data['error']
    assert err['param'] == 'voice'
    assert 'available_voices' in err['details']
    assert 'suggested_defaults' in err['details']
