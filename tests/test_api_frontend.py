from stock_ml_lab.api.frontend import frontend_dist_dir, frontend_is_built


def test_frontend_dist_points_to_project_frontend() -> None:
    path = frontend_dist_dir()
    assert path.name == "dist"
    assert path.parent.name == "frontend"


def test_frontend_build_is_optional() -> None:
    assert isinstance(frontend_is_built(), bool)
