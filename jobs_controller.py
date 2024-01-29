import os
import random

import utils
from classifier import job_classifier


class JobsController:
    JOBS_DIR = os.path.join(".", "data")
    USER_DIR = os.path.join(".", "user")

    def __init__(self, retrain=False):
        self.job_ads = list()

        tot_cat_names = utils.populate_jobs(self.job_ads, JobsController.JOBS_DIR)
        if "user" in os.listdir():
            user_cats = utils.populate_jobs(self.job_ads, JobsController.USER_DIR)
            tot_cat_names.extend(user_cats)
        self.job_ads.sort(key=lambda x: x["webindex"])
        self.cat_names = list()
        self.__update_categories(tot_cat_names)

        self.clf = job_classifier.load_model(
            "job_classifier.mdl", "vectorizer.transformer"
        )

        self.retrain = retrain

    def __update_categories(self, cats: list):
        new_cats = self.cat_names.copy()
        new_cats.extend(cats)
        self.cat_names = sorted(list(set(new_cats)))
        clean_ends = [cat_name.replace(" & ", " ") for cat_name in self.cat_names]
        endpoints = list()
        for cat in clean_ends:
            string_to_take = 2
            done = False
            while not done:
                string_to_take = string_to_take + 1
                candidate = "".join([c[0:string_to_take] for c in cat.split(" ")])
                if candidate not in endpoints:
                    endpoints.append(candidate)
                    done = True
        self.cat_endpoints = ["".join(cat).lower() for cat in endpoints]

    def job_count(self):
        return len(self.job_ads)

    def categories(self):
        return [(n, "section/" + e) for n, e in zip(self.cat_names, self.cat_endpoints)]

    def predict(self, title, description):
        return self.clf.predict(title, description)

    def get_job(self, webindex):
        return [job for job in self.job_ads if job["webindex"] == webindex]

    def add_new_job(self, job_dict: dict):
        job = job_dict.copy()
        # Generate web index
        job["webindex"] = self.get_new_webindex()
        # Save to file. Categories are saved as <arg1>_<arg2> so all spaces need to be replaced
        # We codify & as _ and - as space
        job["category"] = job["category"].strip()
        cls_dir = job["category"].replace(" & ", "_").replace(" ", "-")
        # Create user directory
        if "user" not in os.listdir():
            os.mkdir(JobsController.USER_DIR)
        # Create category directory
        if cls_dir not in os.listdir(JobsController.USER_DIR):
            os.mkdir(os.path.join(JobsController.USER_DIR, cls_dir))
        # Create text to write
        w_out = [
            f"Title: {job['title']}\n",
            f"Webindex: {job['webindex']}\n",
            f"Company: {job['company']}\n",
            f"Description: {' '.join(job['description'].splitlines())}\n",
        ]
        # Files are encoded to bytes
        w_out = [bytes(line, "utf-8") for line in w_out]
        with open(
            os.path.join(JobsController.USER_DIR, cls_dir, f"{job['webindex']}.txt"),
            "wb",
        ) as f:
            f.writelines(w_out)

        # Add job to current running jobs list and categories
        self.job_ads.append(job)
        self.job_ads.sort(key=lambda x: x["webindex"])
        if job["category"] not in self.cat_names:
            self.__update_categories([job["category"]])
        self.__retrain_model()
        return job["webindex"]

    def get_new_webindex(self):
        taken_indices = set([job["webindex"] for job in self.job_ads])
        new_ind = random.randint(10000000, 99999999)
        # Probably ok for a while (776/89999999) combinations
        while new_ind in taken_indices:
            new_ind = random.randint(10000000, 99999999)
        return str(new_ind)

    def get_job_section(self, section):
        section_category = "All available"
        job_section = self.job_ads
        found = True
        if section is not None:
            found = False
            for cat_name, cat_endpoint in zip(self.cat_names, self.cat_endpoints):
                if section == cat_endpoint:
                    found = True
                    section_category = cat_name
                    job_section = [
                        job for job in self.job_ads if job["category"] == cat_name
                    ]
        return job_section, section_category, found

    def __retrain_model(self):
        if self.retrain and self.job_count() % 10 == 0:
            print("Retraining model...")
            self.clf.retrain(self.job_ads)
