<h1>Using AutoML For AI Service Deployment</h1>

<h2>Content</h2>

<br>
<ul>
    <li><a href="#about">About AutoML service</a></li>
    <li><a href="#hot-to-setup">How to setup demo project</a></li>
    <li><a href="#how-to-use">How to use demo project</a></li>
    <li><a href="#precautions">Important notice before using demo project</a></li>
</ul>
<br>

<hr>

<h2 id="about">About AutoML service</h2>

<br>

<p>This demo project presents the idea of AutoML service based on available Python open-source AutoML frameworks.<br>The idea is proposed and explored in our paper titled <b>"Using AutoML for AI service deployment".</b><p>

<p>The main motivation for the project is found in desire to help users with choosing the suitable open-source framework, providing hands-free model training (no coding or machine learning skills required) and deploying the trained model. In addition, we also discuss some other possibilities of the service, like performing benchmark and providing feedback to AutoML developers.</p>

<p style="margin-bottom:0">The key features of the service presented in the paper are:</p>
<ul>
    <li><b>Framework Recommendation System (FRS)</b> - able to suggest suitable framework for the given dataset, ML task and time budget.
    <li><b>Feedback DataBase (FDB)</b> - stores the information about service usages in a publicly available database.<br>This information can be used to further improve FRS, perform benchmark studies and provide feedback to AutoML developers about their framework usage and performances. 
</ul>

<p>This repository is intended to demonstrate the working principle of the proposed idea. As such, it should be considered as a demo project, with only some of the features being implemented in order to illustrate the concept. Please refer to the paper for the in depth details about the project.</p>

<br>
<hr>

<h2 id="hot-to-setup">How to setup demo project</h2>

<br>

<p style="margin-bottom:0">The project application consists of two independent Docker images and shared Docker volume:</p>

<ul>
    <li><b>automl_service_user_side</b> - Handles user interaction in regards to training and using of models.</li>
    <li><b>automl_service_framework_side</b> - Handles the actual training with selected framework.</li>
    <li><b>automl_service_static_content</b> - Shared volume for exchanging files between containers.</li>
</ul>

<p style="margin-bottom:0">To build application, perform following steps:</p>
<ul>
    <li>Ensure the directory structure follows the one in repository:<br>
        <pre style="margin-bottom:0">automl_service/
    ├── user_side/
    ├── framework_side/
    └── docker-compose.yml</pre>
    </li>
    <li>From terminal, cd to directory "automl_service".</li>
    <li>Run following command in terminal:
        <pre style="margin-bottom:0"><code>docker-compose up</code></pre>
    </li>
    <li>The command builds two containers with images, as well as shared volume.
</ul>

<br>
<hr>

<h2 id="how-to-use">How to use demo project</h2>

<br>

<p>The user side container is running on localhost:5000.<br>
The framework side container is running on localhost:5001.</p>

<p style="margin-bottom:0"><b>Train new model:</b></p>
<ul>
    <li>Open in browser both user and framework side.</li> 
    <li>On user side select <b>Train New Model</b> option.</li>
    <li>Upload dataset. The dataset should be a single file in csv format.</li>
    <li>Select task, time budget and target. Enter the name for the model.</li>
    <li>Framework recommendations from FRS are presented. Choose recommended or any other framework. <b>NOTE: For demo purpose FRS is built on simulated data to illustrate the working principle. Actual benchmark needs to be performed, as explained in the paper.</b></li>
    <li>The configuration will be read from framework side. The trained model will be available from <b>Use Trained Model</b> option.
    </li>
</ul>

 <img src="docs/screenshots/user_side_train_model.png" alt="" style="max-width: 90%; margin-left: 5%; margin-right: 5%;">
 
<br>
<p style="margin-bottom:0"><b>Use trained model:</b></p>
<ul>
    <li>Open the user side and select <b>Use Trained Model</b> option.</li>
    <li>Select any previously trained model and choose <b>Load model</b> or <b>Download</b> option.</li>
    <li>If <b>Load model</b> option was chosen, enter new data in appropriate fields and select "Predict" option.</li>
    <li>If <b>Download</b> option was chosen, selected model is saved locally, along with additional information including used framework and packages, task and performance. The model can be imported into other applications (e.g., with joblib).</li>
</ul>

<img src="docs/screenshots/user_side_use_model.png" alt="" style="max-width: 90%; margin-left: 5%; margin-right: 5%;">

<br>
<hr>

<h2 id="precautions">Important notice before using demo project</h2>

<br>

<p>The intention of this demo project is to demonstrate the implementation and usage of AutoML service, as described in our paper.<br> Further development is needed in order to make it fully functional.</p>

<p style="margin-bottom:0">Therefore, please bear in mind following:</p>
<ul>
    <li>Model training by the service is intended only to demonstrate the working principle. Trained models might not be suitable for actual usage.</li>
    <li>Framework recommendation system (FRS) is built on simulated data. The recommendations serve only to demonstrate working principle of FRS.</li>
    <li>Due to early development phase, some issues might be encountered (e.g., various bags and errors, unexpected form of  datasets etc.).</li>
</ul>
