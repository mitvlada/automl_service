import traceback
import logging
import pandas as pd
from flask import Blueprint, render_template, redirect, url_for, jsonify, request, session

from autoML_service.website.train_model.forms import uploadDataForm, autoMLForm, selectFrameworkForm
from autoML_service.autoML.session import autoMLSession
from autoML_service.autoML.functions import set_time_parameter
from autoML_service.autoML.constants import FRAMEWORKS_PARAMETERS
from autoML_service.constants import EXCHANGE_FILE_PATH
from autoML_service.frs.mrf_predictor import MRFPredictor


trainModelViews = Blueprint(
    "trainModelViews",
    __name__,
    static_folder="static",
    static_url_path="/train_model/",
    template_folder="templates",
)


# @trainModelViews.after_request
# def after_request(response):
#     response.headers["Cache-Control"] = "no-cache, must-revalidate"
#     return redirect(url_for("trainModelViews.messageTaskView"))
        

@trainModelViews.route("/train/configure_task", methods=["GET", "POST"])
def configureTaskView(): 

    upload_form = uploadDataForm()
    configure_form = autoMLForm()
    select_framework_form = selectFrameworkForm()

    data_uploaded = False
    task_configured = False
    rankings = None

    autoMLSession.is_configured = False

    if request.form:

        if upload_form.validate_on_submit():
            dataset = upload_form.dataset.data

            filename = dataset.filename
            dataset = pd.read_csv(dataset)
            columns = dataset.columns.values.tolist()
            configure_form.target.choices = [(column, column) for column in columns]
            
            try:
                dataset.to_csv("".join([EXCHANGE_FILE_PATH, filename]))
                print("Data saved???")
            except:
                print("Data not saved!!!")
            
            autoMLSession.add_session_parameter(filename, "filename")
            autoMLSession.add_session_parameter(columns, "columns")

            upload_form = upload_form
            data_uploaded = True
    
        elif configure_form.validate_on_submit():
            name = configure_form.name.data
            task = configure_form.task.data
            target = configure_form.target.data
            time_budget = configure_form.time.data

            autoMLSession.add_session_parameter(name, "name")
            autoMLSession.add_session_parameter(task, "task")
            autoMLSession.add_session_parameter(target, "target")
            autoMLSession.add_session_parameter(time_budget, "time_budget")

            mrf = MRFPredictor(autoMLSession.parameters["task"])
            mrf.get_ranked_predictions(autoMLSession.parameters["filename"], autoMLSession.parameters["time_budget"])
            
            rankings = mrf.df_rankings.to_html(index=False, border=0)        
            
            # set caption in rankings html
            line = '<thead>'
            index = rankings.find(line)
            rankings = rankings[:index] + '<caption>Framework recommendations</caption>\n' + rankings[index:]

            upload_form = upload_form
            configure_form = configure_form
            data_uploaded = True
            task_configured = True

        elif select_framework_form.validate_on_submit():
            try:
                framework = select_framework_form.autoML.data
                parameter, unit = set_time_parameter(autoMLSession.parameters["time_budget"], framework)
                framework_parameters = FRAMEWORKS_PARAMETERS[framework]
                framework_parameters[parameter] = unit
                
                autoMLSession.add_session_parameter(framework, "framework") 
                autoMLSession.add_session_parameter(framework_parameters, "parameters")         

                autoMLSession.is_configured = True
            except Exception as e:
                logging.error(traceback.format_exc())
                print("ERROR")

            return redirect(url_for("trainModelViews.taskConfiguredView"))
            # return redirect(url_for("trainModelViews.messageTaskView"))
        
    return render_template("configure.html", 
                           upload_form=upload_form, 
                           configure_form=configure_form, 
                           select_framework_form=select_framework_form, 
                           data_uploaded=data_uploaded,
                           task_configured=task_configured,
                           rankings=rankings
                           )

# @trainModelViews.route(f"/train/configure", methods=["GET", "POST"])
# def configAutoMLView():
#     upload_form = uploadDataForm()
#     configure_form = autoMLForm()
#     configure_form.target.choices = autoMLSession.parameters["columns"]

#     data_uploaded=True

#     if configure_form.validate_on_submit():
#         framework = configure_form.autoML.data
#         task = configure_form.task.data
#         target = configure_form.target.data

#         autoMLSession.add_session_parameter(framework, "framework")
#         autoMLSession.add_session_parameter(task, "task")
#         autoMLSession.add_session_parameter(target, "target")
        
#         return redirect(url_for("trainModelViews.messageView")) 
#         # return redirect(url_for("trainModelViews.uploadDataView")) 

#     render_template("upload.html", upload_form=upload_form, configure_form=configure_form, data_uploaded=data_uploaded)
#     # return render_template("configure.html", form=configure_form)


@trainModelViews.route("/train/task_configured")
def taskConfiguredView():
    if autoMLSession.is_configured:
        return render_template("task_configured.html")
    else:
        return redirect(url_for("trainModelViews.configureTaskView"))


@trainModelViews.route("/message")
def messageTaskView():
    session_params = autoMLSession.parameters
    
    return jsonify(session_params)
    