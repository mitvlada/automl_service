from flask_wtf import FlaskForm
from wtforms.fields import FileField, SelectField, IntegerField, SubmitField, StringField
from wtforms.validators import DataRequired, length, NumberRange

from autoML_service.autoML.constants import AUTOML_CHOICES

class uploadDataForm(FlaskForm):

    dataset = FileField(validators=[DataRequired()])
    submit = SubmitField("Upload")


class autoMLForm(FlaskForm):

    ml_tasks = [
		("classification", "classification"),
		("regression", "regression")
	]
	
    # autoML = SelectField("Framework", choices=AUTOML_CHOICES, validators=[DataRequired()])
	
	
    name = StringField("Name", validators=[DataRequired(), length(max=50)])
    target = SelectField("Target", choices=[], validate_choice=False, validators=[DataRequired()])

    task = SelectField("Task", choices=ml_tasks, validators=[DataRequired()])
	
    time = IntegerField("Time budget", validators=[DataRequired(), NumberRange(min=1, max=240, message='Time budget')], render_kw={"placeholder": "Minutes"})
    submit = SubmitField("Configure")
	

class selectFrameworkForm(FlaskForm):

    autoML = SelectField("Select Framework", choices=AUTOML_CHOICES, validators=[DataRequired()])
    submit = SubmitField("Start training")
	

class configureAutoMLForm(FlaskForm):
    """ PLACEHOLDER """
    pass