# Flask webapp with NL model integration

## Requirements
- flask
- scikit-learn
- nltk

---

## Usage

To run the app use

```
python app.py
```

Optionally you may add arguments to the command

    `--bind`:    Binds the server to the default port (5000) on the loopback IP
                 so that it is visible over a local network
    `--retrain`: Sets the server to retrain its model every 10 job entries.
                 * This is disabled by default as it takes a little bit of
                   extra time to retrain a model and Flask has no good way
                   to reactively disable buttons.
                 * This will also affect the accuracy of the classifier as new
                   categories will have much less data and would not be likely
                   to be predicted until more data a retrieved.

## Structure

### Directory Structure

This web app runs under the assumption of job information being in the
directory format of `/<data | user>/<category>/<job info>` and that all
web indexes are unique. The `data` directory contains the same data from
milestone 1 to populate the job board. The `user` directory is created as
needed when a job post is made. A job listing must be in the specific format
(newline separated, `Title: ` and `Description: ` keys).

When posting a job, users may click on the classify button to classify their
job entry. This requires a title and description. Attempting to submit without
a category will lead to an automatic classification as the category is pivotal
to the data structure.

Users may ignore the classifier and enter custom categories. This will be
reflected in the categories navbar and the endpoint will automatically update.

If the model is set to retrain, the vectorizer and model will be replaced when
the update occurs.

If you want the original model, this can be found in the `defaults` directory.
It is also safe to delete the `user` directory to reset the job ads and
categories.

### Code Structure

The app uses a controller to handle the job data. This is so the main app file
is less messy due to the high amount of mapping parameters and the logic of the
app is contained together. The classification is handled by the classifier
which handles the model, which the controller will make calls to. The
classifier handles the data processing so that it can be used by the model.
