from flask import (Flask, redirect, render_template, request,
                   send_from_directory, session)

from jobs_controller import JobsController
from utils import get_back_info

app = Flask(__name__)

ctl = JobsController()


@app.route("/create", methods=["GET", "POST"])
def create_page():
    back_section, page_query = get_back_info(session)

    if request.method == "POST":
        if request.form["button"] == "submit" and request.form["category"] != "":
            title = request.form["title"]
            company = request.form["company"]
            description = request.form["description"]
            category = request.form["category"]
            job = {
                "title": title,
                "company": company,
                "description": description,
                "category": category,
            }
            webindex = ctl.add_new_job(job)

            return redirect("/job/" + webindex)

        title = request.form["title"]
        company = request.form["company"]
        description = request.form["description"]
        show_error = request.form["button"] != "classify"
        

        categories = ctl.predict(title, description)
        category = categories[0][0]

        return render_template(
            "create-job.html",
            global_cats=ctl.categories(),
            page_title="Job Posting",
            forgot_category=show_error,
            job_info={
                "title": title,
                "company": company,
                "description": description,
                "category": category,
            },
            cc=categories,
            section=back_section,
            page=page_query,
        )
    else:
        title = ""
        company = ""
        description = ""
        category = ""

        return render_template(
            "create-job.html",
            global_cats=ctl.categories(),
            page_title="Job Posting",
            forgot_category=False,
            job_info={
                "title": title,
                "company": company,
                "description": description,
                "category": category,
            },
            cc=list(),
            section=back_section,
            page=page_query,
        )


@app.route("/job/<webindex>")
def listing(webindex):
    back_section, page_query = get_back_info(session)

    job = ctl.get_job(webindex)
    if job:
        return render_template(
            "display-job.html",
            global_cats=ctl.categories(),
            page_title="Job Details",
            job=job[0],
            section=back_section,
            page=page_query,
        )
    return redirect("/?webindex-failed=true")


@app.route("/")
@app.route("/section/<section>")
def index(section=None):
    job_section, section_category, found = ctl.get_job_section(section)

    webindex_not_found = request.args.get("webindex-failed")
    if webindex_not_found:
        found = False

    page_number = request.args.get("page")
    # No param is first page
    if page_number is None:
        page_number = 0
    #  Try to typecast, on failure just go home
    try:
        page_number = int(page_number)
    except ValueError:
        page_number = 0

    if section != "favicon.ico":
        session["section"] = section
        session["page"] = page_number

    jobs_per_page = 10
    jobs_start_index = page_number * jobs_per_page
    jobs_end_index = jobs_start_index + jobs_per_page
    total_jobs = len(job_section)
    if jobs_end_index > total_jobs:
        jobs_end_index = total_jobs
    return render_template(
        "index.html",
        global_cats=ctl.categories(),
        page_title=section_category,
        endpoint_not_exist=not found,
        section_info={
            "title_text": section_category + " jobs",
            "section": "section/" + session["section"] if session["section"] else "",
        },
        jobs_display=job_section[jobs_start_index:jobs_end_index],
        pagination_info={
            "total": total_jobs,
            "page_num": page_number,
            "index_start": jobs_start_index,
            "index_end": jobs_end_index,
            "jobs_per_page": jobs_per_page,
        },
        max_length=500,
    )


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico")


app.secret_key = "meow"

if __name__ == "__main__":
    import sys

    if "--retrain" in sys.argv:
        print("Model will update itself.")
        ctl.retrain = True
    if "--bind" in sys.argv:
        print("App is available on local network")
        app.run(host="0.0.0.0")
    else:
        app.run()
