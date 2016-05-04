from flask import Flask


def create_app():
    app = Flask(__name__, static_folder="../static", static_url_path="/static")

    from api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    app.debug = True

    return app
