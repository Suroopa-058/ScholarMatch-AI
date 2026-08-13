def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_recommendation_structure_and_top_five(client, valid_profile):
    response = client.post("/api/recommend", json=valid_profile)
    body = response.json()
    assert response.status_code == 200
    assert body["student_id"] == 12345
    assert len(body["recommendations"]) == 5
    assert set(body["recommendations"][0]) == {"rank", "scholarship_id", "name", "description", "recommendation_score", "semantic_similarity", "major_match", "eligible", "academic_fit"}
    assert [item["rank"] for item in body["recommendations"]] == [1, 2, 3, 4, 5]


def test_invalid_gpa(client, valid_profile):
    valid_profile["gpa"] = 11
    response = client.post("/api/recommend", json=valid_profile)
    assert response.status_code == 422
    assert "gpa" in response.json()["errors"]


def test_invalid_student_profile(client, valid_profile):
    del valid_profile["class"]
    assert client.post("/api/recommend", json=valid_profile).status_code == 422


def test_exactly_ten_internal_candidates(client, valid_profile):
    service = client.app.state.recommendation_service
    features, context = service.candidate_features(type("Profile", (), {**valid_profile, "class_": valid_profile["class"]})())
    assert features.shape == (10, 11)
    assert len(context) == 10


def test_recommend_then_explain(client, valid_profile):
    recommendation = client.post("/api/recommend", json=valid_profile).json()["recommendations"][0]
    response = client.post("/api/explain", json={"student_id": 12345, "scholarship_id": recommendation["scholarship_id"]})
    assert response.status_code == 200
    assert response.json()["explanation"][0]["feature"] == "GPA"


def test_unknown_scholarship(client):
    response = client.post("/api/explain", json={"student_id": 12345, "scholarship_id": "SCH999"})
    assert response.status_code == 404
