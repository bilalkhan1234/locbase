from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, BooleanField

from wtforms import validators, ValidationError
from wtforms.validators import DataRequired

class Registro(FlaskForm):
    name = TextField('Profession', [validators.Required('Name required')])
    submit = SubmitField('Submit')