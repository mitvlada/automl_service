import os
from flask_wtf import FlaskForm
from wtforms.fields import SubmitField, IntegerField, DecimalField, SelectField
from wtforms.validators import InputRequired, DataRequired

from autoML_service.constants import MODELS_FILE_PATH


def createGetModelForm():
    class getModelForm(FlaskForm):
                
        trained_models = os.listdir(MODELS_FILE_PATH)
        trained_models.sort()
        trained_models_choices = [(file, file) for file in trained_models if file.endswith(".automl")]

        model = SelectField("Trained model", choices=trained_models_choices, validators=[DataRequired()])       
        submit = SubmitField("Load model")
        download = SubmitField("Download model")

    return getModelForm()


def createModelFeaturesForm(model):
    class modelFeaturesForm(FlaskForm):
        pass

    features = []
    if model is not None:
        if 'features' in model:
            for key, value in model['features'].items():
                if value['type'] == "int64":
                    setattr(modelFeaturesForm, key, IntegerField(validators=[InputRequired()], render_kw={"placeholder": "Integer"}))
                elif value['type'] == "float64":
                    setattr(modelFeaturesForm, key, DecimalField(validators=[InputRequired()], render_kw={"placeholder": "Float"}))
                elif value['type'] == "object":
                    choices = [(val, key) for key, val in value['mapping'].items()]

                    # Default coerce method is 'str'.
                    # Setting coerce=int makes the form fail on validation if default choice has been selected.
                    # Value is currently converted to float in the views, but see what is the issue with coerce.
                    setattr(modelFeaturesForm, key, SelectField(choices=choices, validators=[InputRequired()], render_kw={"placeholder": "Object"}))

                features.append(key)  # What if there is no feature has been set?
    
    setattr(modelFeaturesForm, "submit", SubmitField("Predict"))
    setattr(modelFeaturesForm, "features", features)

    return modelFeaturesForm()
