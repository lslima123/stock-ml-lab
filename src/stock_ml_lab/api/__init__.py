__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from stock_ml_lab.api.app import app
        return app
    raise AttributeError(name)
