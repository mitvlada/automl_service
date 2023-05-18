from flask import Flask
from autoML_service.autoML.session import autoMLSession

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "asfaafasg"

    from autoML_service.website.base.views import baseViews
    from autoML_service.website.train_model.views import trainModelViews
    from autoML_service.website.use_model.views import useModelViews

    app.register_blueprint(baseViews, url_prefix="/")
    app.register_blueprint(trainModelViews, url_prefix="/")
    app.register_blueprint(useModelViews, url_prefix="/")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
