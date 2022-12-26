import copy
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None, 'display.width', -1)
gun_violence_df = pd.read_csv('gun-violence-data_01-2013_03-2018.csv')

gun_violence_df = gun_violence_df.dropna(subset='n_killed')

gun_violence_df = gun_violence_df.dropna(subset='n_injured')

gun_violence_df.drop_duplicates()

gv_df_clean = gun_violence_df.drop(columns=['incident_url', 'source_url', 'incident_url_fields_missing',
                                            'notes', 'participant_name', 'participant_type',
                                            'sources', 'gun_type', 'participant_age_group'])

gv_df_clean = gv_df_clean.dropna(subset=['state', 'participant_age', 'participant_gender',
                                         'date', 'state_house_district', 'state_senate_district',
                                         'congressional_district', 'latitude', 'longitude', ])

gv_df_clean = gv_df_clean.drop(columns=['gun_stolen', 'address', 'incident_characteristics', 'location_description'])

gv_df_clean = gv_df_clean.drop(columns=['n_guns_involved', 'participant_relationship', ])

gv_df_clean['is_killed'] = gv_df_clean.apply(lambda row: 1.0 if row['n_killed'] > 0 else 0.0, axis=1)

gv_df_clean['is_injured'] = gv_df_clean.apply(lambda row: 1.0 if row['n_injured'] > 0 else 0.0, axis=1)

# gv_df_clean['date'] = pd.to_datetime(gv_df_clean['date'])
#
# print(gv_df_clean['date'].year)
gv_df_clean['year'] = gv_df_clean.apply(lambda row: float(pd.to_datetime(row['date']).year), axis=1)

gv_df_clean['month'] = gv_df_clean.apply(lambda row: float(pd.to_datetime(row['date']).month), axis=1)

gv_df_clean['day'] = gv_df_clean.apply(lambda row: float(pd.to_datetime(row['date']).day), axis=1)

print(gv_df_clean)

city_codes = {}


def add_city_code(city_str):
    if city_codes.get(city_str) is not None:
        return city_codes[city_str]
    else:
        city_codes[city_str] = len(city_codes)
        return city_codes[city_str]


gv_df_clean['city_code'] = gv_df_clean.apply(lambda row: float(add_city_code(row['city_or_county'])), axis=1)


def parse_age(participant_age_str):
    ages_str = participant_age_str.replace('||', '|').split('|')
    ages = []
    for age_str in ages_str:
        right = age_str.replace('::', ':').split(':')
        ages.append(int(right[1]))
    sum = 0
    for i in range(0, len(ages)):
        sum += ages[i]
    return sum / len(ages)


def parse_gender(part_gender_str):
    genders_str = part_gender_str.replace('||', '|').split('|')
    genders = []
    for gender_str in genders_str:
        right = gender_str.replace('::', ':').split(':')
        genders.append(right[1])

    n_male = 0
    n_female = 0
    for i in range(0, len(genders)):
        if genders[i] == 'Male':
            n_male += 1
        elif genders[i] == 'Female':
            n_female += 1
    return 'Male' if n_male > n_female else 'Female'


gv_df_clean['average_age'] = gv_df_clean.apply(lambda row: float(parse_age(row['participant_age'])), axis=1)
gv_df_clean['average_gender'] = gv_df_clean.apply(lambda row: parse_gender(row['participant_gender']), axis=1)

gv_df_clean = gv_df_clean.drop(
    columns=['participant_age', 'participant_gender', 'participant_status', 'n_killed', 'n_injured'])

state_names = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
               "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana",
               "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
               "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
               "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
               "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
               "Washington", "West Virginia", "Wisconsin", "Wyoming"]


def find_state_code(state):
    for i in range(0, len(state_names)):
        if state.upper().replace(' ', '') == state_names[i].upper().replace(' ', ''):
            return i


gv_df_clean['state_code'] = gv_df_clean.apply(lambda row: float(find_state_code(row['state'])), axis=1)
gv_df_clean['gender_bool'] = gv_df_clean.apply(lambda row: 1.0 if row['average_gender'] == 'Male' else 0.0, axis=1)
gv_df_clean['is_injured_or_killed'] = gv_df_clean.apply(
    lambda row: 1.0 if row['is_killed'] or row['is_injured'] else 0.0,
    axis=1)

gv_ones = gv_df_clean[gv_df_clean['is_injured_or_killed'] != 0.0].sample(frac=0.5, random_state=12345)

gv_df_clean = pd.concat([gv_ones] + [gv_df_clean[gv_df_clean['is_injured_or_killed'] == 0.0]])

filter_age = gv_df_clean['average_age'] < 80.0
filter_age_2 = gv_df_clean['average_age'] > 13.0
gv_df_clean = gv_df_clean[filter_age & filter_age_2]

from sklearn.utils import shuffle

gv_df_clean = shuffle(gv_df_clean)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_columns = ['state_code', 'average_age', 'gender_bool',
                'year', 'month', 'day',
                'city_code', 'is_injured', 'state_house_district',
                'state_senate_district', 'congressional_district',
                'latitude', 'longitude']

df_target = gv_df_clean.is_injured_or_killed
df_data = gv_df_clean[data_columns]
x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.25, random_state=125)


class MyLogisticRegression:

    def __init__(self):

        self.bias = None
        self.weights = None

    def fit(self, x_train, y_train, learning_rate, num_iterations):

        x = self.normalize(
            np.array(
                copy.deepcopy(
                    x_train
                )
            )
        )

        y = np.array(
            copy.deepcopy(
                y_train
            )
        )

        self.weights, self.bias = self.init_w_b(
            size=x_train.shape[0],
            learning_rate=learning_rate
        )

        losses = []

        for i in range(num_iterations):

            loss, dw, db = self.grad_descend(
                x_train=x,
                y_train=y
            )

            losses.append(loss)

            self.update_w_b(
                dw=dw,
                db=db,
                learning_rate=learning_rate,
            )

            if i % 10 == 0:
                print(f"Loss on {i}: {loss}")

    def grad_descend(self, x_train, y_train):

        z = np.dot(self.weights.T, x_train) + self.bias

        y_head = self.sigmoid(z)

        one_losses = -y_train * np.log(y_head)
        zero_losses = (1 - y_train) * np.log(1 - y_head)

        loss_diff = one_losses - zero_losses
        loss = (np.sum(loss_diff)) / x_train.shape[1]

        dw = (np.dot(x_train, (y_head - y_train).T)) / x_train.shape[1]
        db = np.sum(y_head - y_train) / x_train.shape[1]

        return loss, dw, db

    def predict_proba(self, x_test):

        x = self.normalize(
            np.array(
                copy.deepcopy(
                    x_test
                )
            )
        )

        z = self.sigmoid(
            z=np.dot(self.weights.T, x) + self.bias
        )
        return z

    def predict(self, x_test):

        x = self.normalize(
            np.array(
                copy.deepcopy(
                    x_test
                )
            )
        )

        z = self.sigmoid(
            z=np.dot(self.weights.T, x) + self.bias
        )

        predict = np.zeros((1, x.shape[1]))

        for i in range(z.shape[1]):

            if z[0, i] <= 0.5:
                predict[0, i] = 0
            else:
                predict[0, i] = 1

        return predict

    def accuracy(self, y_test, predict):

        y = np.array(
            copy.deepcopy(
                y_test
            )
        )

        print(f"test accuracy: {100 - np.mean(np.abs(predict - y)) * 100}")

    def init_w_b(self, size, learning_rate):

        return np.full((size, 1), learning_rate), 0.0

    def update_w_b(self, dw, db, learning_rate):

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

    def normalize(self, x):

        for i in range(x.shape[0]):
            x[i] = (x[i] - x[i].mean(axis=0)) / x[i].std(axis=0)

        return x

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))


class Metrics:

    def __init__(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0
        self.precision_positive = 0
        self.precision_negative = 0
        self.recall_positive = 0
        self.recall_negative = 0

    def confusion_matrix(self, actual_values, predicted_values):

        for actual_value, predicted_value in zip(actual_values, predicted_values):

            if predicted_value == actual_value:
                if predicted_value == 1:
                    self.tp += 1
                else:
                    self.tn += 1

            else:
                if predicted_value == 1:
                    self.fp += 1
                else:
                    self.fn += 1

        return [
            [self.tn, self.fp],
            [self.fn, self.tp]
        ]

    def accuracy(self, predicted_values):
        return (self.tp + self.tn) / predicted_values.shape[0]

    def precision_recall_score(self):
        all_predicted_positives = self.tp + self.fp
        self.precision_positive = self.tp / all_predicted_positives

        all_predicted_negatives = self.tn + self.fn
        self.precision_negative = self.tn / all_predicted_negatives

        all_actual_positive = self.tp + self.fn
        self.recall_positive = self.tp / all_actual_positive

        all_actual_negative = self.tn + self.fp
        self.recall_negative = self.tn / all_actual_negative
        return [
            [self.precision_positive, self.precision_negative],
            [self.recall_positive, self.recall_negative]
        ]

    def f1_score(self):
        f1_pos = 2 * (self.precision_positive * self.recall_positive) \
                 / (self.precision_positive + self.recall_positive)

        f1_neg = 2 * (self.precision_negative * self.recall_negative) \
                 / (self.precision_negative + self.recall_negative)
        return [f1_pos, f1_neg]

    def print_cnf(self):
        class_names = [0, 1]

        fig, ax = plt.subplots()

        tick_marks = np.arange(len(class_names))

        plt.xticks(tick_marks, class_names)

        plt.yticks(tick_marks, class_names)

        matr = np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])

        sns.heatmap(
            pd.DataFrame(matr),
            annot=True,
            cmap="YlGnBu",
            fmt='g'
        )

        ax.xaxis.set_label_position("top")

        plt.tight_layout()

        plt.title('Матрица ошибок', y=1.1)

        plt.ylabel('Ответы')

        plt.xlabel('Предсказания')
        plt.show()

    def print_pr_curve(self, y_test, probs):
        pr_scores = []
        recall_scores = []

        prob_thresholds = np.linspace(0, 1, num=100)

        for p in prob_thresholds:

            y_test_preds = []

            for prob in probs:
                if prob > p:
                    y_test_preds.append(1)
                else:
                    y_test_preds.append(0)

            pres, rec = self.calc_pr_rec_for_proba(y_test, y_test_preds)
            if pres == float('inf'):
                break
            pr_scores.append(pres)
            recall_scores.append(rec)

        self.print_curve(pr_scores, recall_scores)

    def print_curve(self, precisions, recalls):
        fig, ax = plt.subplots()
        ax.plot(recalls, precisions, color='purple')

        ax.set_title('Кривая точности-полноты')
        ax.set_ylabel('Точность')
        ax.set_xlabel('Полнота')

        plt.show()

    def calc_pr_rec_for_proba(self, y_test, y_preds):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for actual_value, predicted_value in zip(y_test, y_preds):

            if predicted_value == actual_value:
                if predicted_value == 1:
                    tp += 1
                else:
                    tn += 1

            else:
                if predicted_value == 1:
                    fp += 1
                else:
                    fn += 1

        all_predicted_positives = tp + fp
        if all_predicted_positives != 0:
            precision_positive = tp / all_predicted_positives
            all_actual_positive = tp + fn
            recall_positive = tp / all_actual_positive
            return precision_positive, recall_positive
        else:
            return float('inf'), float('inf')


if __name__ == '__main__':
    lr = MyLogisticRegression()
    time_bef = time.time()
    lr.fit(
        x_train=x_train.T,
        y_train=y_train.T,
        learning_rate=0.01,
        num_iterations=1000
    )
    time_after = time.time()
    print("!" * 40)
    print(f"Время обучения: {time_after - time_bef}")
    pr = lr.predict(x_test=x_test.T)
    print("!" * 40)
    print(f"Время классификации: {time.time() - time_after}")
    lr.accuracy(
        y_test=y_test.T,
        predict=pr
    )
    proba = lr.predict_proba(x_test=x_test.T)

    metr = Metrics()
    cnf = metr.confusion_matrix(y_test, pr.T)

    metr.print_cnf()
    print(f"accur: {metr.accuracy(y_test)}")
    pr_matrics = metr.precision_recall_score()
    print(f"f1: {metr.f1_score()}")
    metr.print_pr_curve(y_test=y_test, probs=proba.T)
