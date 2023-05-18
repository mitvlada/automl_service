from flask import Blueprint, render_template, redirect, url_for, jsonify, request


baseViews = Blueprint(
    "baseViews",
    __name__,
    static_folder="static",
    static_url_path="/base/",
    template_folder="templates",
)


@baseViews.route("/")
def homeView():
    return render_template("home.html")


@baseViews.route("/about")
def aboutView():
    return render_template("about.html")
