import itertools

import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


class DataCleaner:
    def __init__(self):
        self.tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        self.stopwords = set([word.strip() for word in open("stopwords_en.txt", "r")])

    def tokenize(self, description):
        sentences = nltk.tokenize.sent_tokenize(description.lower())
        token_lists = [self.tokenizer.tokenize(sen) for sen in sentences]
        return list(itertools.chain.from_iterable(token_lists))

    def filter_on(self, tokens, key):
        return [w for w in tokens if key(w)]

    def apply_filter(self, document_tks, key):
        return [self.filter_on(d, key) for d in document_tks]

    def shallow_clean(self, title, description):
        """We don't need to fully clean this data as its just one document"""
        tks = self.tokenize(description)
        tks = self.filter_on(tks, lambda w: len(w) > 1 and w not in self.stopwords)
        return [" ".join(tks) + " " + title]

    def __get_once_words(self, tokenized_descriptions):
        all_tks = list(itertools.chain.from_iterable(tokenized_descriptions))
        term_fd = nltk.probability.FreqDist(all_tks)
        return term_fd.hapaxes()

    def __get_most_words(self, tokenized_descriptions):
        unique_tokens = list(
            itertools.chain.from_iterable([set(d) for d in tokenized_descriptions])
        )
        doc_fd = nltk.probability.FreqDist(unique_tokens)
        return [w[0] for w in doc_fd.most_common(50)]

    def deep_clean(self, descriptions):
        """Does all the cleaning from A2 Task 1"""
        tk_desc = [self.tokenize(d) for d in descriptions]
        tk_desc = self.apply_filter(
            tk_desc, lambda w: len(w) > 1 and w not in self.stopwords
        )
        once_words = self.__get_once_words(tk_desc)
        most_words = self.__get_most_words(tk_desc)
        extreme_words = set(once_words + most_words)
        tk_desc = self.apply_filter(tk_desc, lambda w: w not in extreme_words)
        return tk_desc


class JobClassifier:
    def __init__(self, model, transformer):
        self.model = model
        self.vectorizer = transformer
        self.__cleaner = DataCleaner()

    def predict(self, title, description):
        corpus = self.__cleaner.shallow_clean(title, description)
        features = self.vectorizer.fit_transform(corpus)
        pred_probs = self.model.predict_proba(features)[0]
        mean_prob = 1.0 / len(pred_probs)
        given_prob = sum([p for p in pred_probs if p > mean_prob])
        predictions = [
            (self.targets()[i].replace("_", " & "), p / given_prob)
            for i, p in enumerate(pred_probs)
            if p > mean_prob
        ]
        predictions.sort(reverse=True, key=lambda x: x[1])
        return predictions

    def targets(self):
        return self.model.classes_

    def __evaluate(self, x_train, x_test, y_train, y_test):
        model = LogisticRegression(max_iter=10000)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test), model

    def __kfold_select(self, features, targets, splits=5):
        results = list()
        for train_i, test_i in KFold(n_splits=splits, shuffle=True).split(
            range(0, len(targets))
        ):
            print(".", end="")
            y_train = [str(targets[i]) for i in train_i]
            y_test = [str(targets[i]) for i in test_i]
            results.append(
                self.__evaluate(features[train_i], features[test_i], y_train, y_test)
            )
        print()
        return results

    def retrain(self, jobs):
        descriptions = list()
        targets = list()
        titles = list()
        for job in jobs:
            descriptions.append(job["description"])
            targets.append(job["category"])
            titles.append(job["title"])
        print("Cleaning descriptions")
        doc_tokens = self.__cleaner.deep_clean(descriptions)
        vocab = sorted(list(set(itertools.chain.from_iterable(doc_tokens))))
        print("Creating new vectorizer")
        self.vectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocab)
        documents = [
            " ".join(desc) + " " + title for desc, title in zip(doc_tokens, titles)
        ]
        features = self.vectorizer.fit_transform(documents)
        print("Selecting model", end="")
        results = self.__kfold_select(features, targets, splits=3)
        results.sort(key=lambda x: x[0])
        self.model = results[len(results) // 2][1]
        print(f"New model is {results[len(results) // 2][0] * 100:.2f}% accurate.")
        print("Dumping model and transformer")
        dump_model(
            self.model, "job_classifier.mdl", self.vectorizer, "vectorizer.transformer"
        )


def load_model(model_fn, v_fn):
    return JobClassifier(joblib.load(model_fn), joblib.load(v_fn))


def dump_model(model, model_fn, v, v_fn):
    joblib.dump(model, model_fn)
    joblib.dump(v, v_fn)
