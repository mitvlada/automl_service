import pandas as pd
from flask import (
    Blueprint,
    render_template,
    flash,
    redirect,
    url_for,
    request,
    send_from_directory,
)
import joblib
import logging
import traceback
import numpy as np
from pathlib import Path

from autoML_service.website.use_model.forms import (
    createGetModelForm,
    createModelFeaturesForm,
)
from autoML_service.constants import EXCHANGE_FILE_PATH, MODELS_FILE_PATH
from autoML_service.autoML.constants import LIGHTAUTOML


useModelViews = Blueprint(
    "useModelViews",
    __name__,
    static_folder="static",
    static_url_path="/use_model/",
    template_folder="templates",
)


@useModelViews.route("/use/get_model", methods=["GET", "POST"])
def getModelView():

    models_path = Path(MODELS_FILE_PATH)

    if not models_path.is_dir():
        models_exist = False
        return render_template("get_model.html", form=None, models_exist=models_exist)

    models_exist = True
    form = createGetModelForm()

    if form.validate_on_submit():

        model_name = form.model.data
        filepath = "".join([MODELS_FILE_PATH, model_name])

        if Path.is_file(Path(filepath)):
            if "submit" in request.form:
                return redirect(url_for("useModelViews.useModelView", filepath=filepath))
            elif "download" in request.form:
                return send_from_directory(MODELS_FILE_PATH, model_name, as_attachment=True)
        else:
            flash("Model not found.")

    return render_template("get_model.html", form=form, models_exist=models_exist)


@useModelViews.route("/use/use_model", methods=["GET", "POST"])
def useModelView():

    filepath = request.args.get("filepath")
    if filepath is None:
        return redirect(url_for("useModelViews.getModelView"))
    elif not Path.is_file(Path(filepath)):
        return redirect(url_for("useModelViews.getModelView"))

    model_data = joblib.load(filepath)
    model = model_data["model"]
    form = createModelFeaturesForm(model_data)

    if form.validate_on_submit():

        # WTForms' DecimalField returns number as decimal.Decimal which doesn't work with AutoSklearn.
        # Also, numpy array can have only one data type.
        # For that reason, everything is converted to float.
        new_data = [
            float(field.data)
            for field in form
            if field.name not in ["submit", "csrf_token"]
        ]

        if model_data["framework"] == LIGHTAUTOML:
            predictions = model.predict(
                pd.DataFrame(
                    columns=list(model_data["features"].keys()), data=[new_data]
                )
            ).data

            if model_data["task"] == "classification":
                predictions = [str(pred[0]) + " %" for pred in predictions]
            else:
                predictions = np.reshape(predictions, len(predictions))

        else:
            predictions = model.predict(np.array(new_data).reshape(1, -1))

        target_predictions = {
            target: prediction
            for target, prediction in zip(model_data["classes"], predictions)
        }
        return render_template(
            "use_model.html",
            form=form,
            model=model_data,
            target_predictions=target_predictions,
        )

    return render_template(
        "use_model.html", form=form, model=model_data, target_predictions=None
    )
