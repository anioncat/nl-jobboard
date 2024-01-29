import html

from sklearn.datasets import load_files


def populate_jobs(master_list, path):
    descriptors = ["title", "webindex", "company", "description"]
    # & characters are missing from html decode
    html_replacers = {
        "nbsp;": "&nbsp;",
        "ndash;": "&ndash;",
        "quot;": "&quot;",
        "rsquo;": "&rsquo;",
    }
    data_load = load_files(path)
    data = [job.decode("utf-8") for job in data_load.data]
    for w, i in html_replacers.items():
        data = [job.replace(w, i) for job in data]
    jobs_info = [
        (html.unescape(job).split("\n"), target)
        for job, target in zip(data, data_load.target)
    ]
    cat_names = [
        name.replace("_", " & ").replace("-", " ") for name in data_load.target_names
    ]
    for job, target in jobs_info:
        job_ad = dict()
        job_ad["category"] = cat_names[target]
        for cat in job:
            for descriptor in descriptors:
                starter = f"{descriptor}: "
                if cat.lower().startswith(starter):
                    job_ad[descriptor] = cat[len(starter) :]
        master_list.append(job_ad)
    return cat_names

def get_back_info(session):
    back_section = session["section"]
    if not back_section:
        back_section = ""
    else:
        back_section = "section/" + back_section
    back_page = session["page"]
    if not back_page:
        back_page = "0"
    if back_page == "0":
        page_query = ""
    else:
        page_query = f"?page={back_page}"
    return back_section, page_query
