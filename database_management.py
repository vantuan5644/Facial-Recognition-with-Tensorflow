import datetime
import random
import time
import uuid
import warnings

import pandas as pd


def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    return df.sort_index()


def find_someone_info(what_to_find, what_you_had, value, db):
    # db is a pandas dataframe
    had_that_column = False
    for i in db.columns:
        if i == what_you_had:
            had_that_column = True
    if not had_that_column:
        warnings.warn(f"There is no column in given database: {what_you_had}")
    if had_that_column:
        if not (value in db[what_you_had].values):
            warnings.warn(f"Can't find value: {value} in given database")
        else:
            # found that value
            match_row = db.loc[db[what_you_had] == value]
            match_info = match_row[what_to_find].values[0]
            return match_info


class writeToCsv:
    def __init__(self, local_db_path='./database/database.csv', event_log_path='./database/event_log.csv',
                 time_interval=10):

        self.local_db_path = local_db_path
        self.event_log_path = event_log_path

        self.local_db = pd.read_csv(local_db_path)
        self.event_log = pd.read_csv(event_log_path)
        self.time_interval = time_interval
        self.time_start = time.monotonic()
        self.name = None

    def add_new_user(self, name, df=None):

        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        menu = ['Caramel Macchiato', 'Tra Oolong Vai Nhu Y', 'Tra Matcha Macchiato', 'Tra Den Macchiato',
                'Tra Dao Cam Sa', 'Ca Phe Den', 'Americano', 'Bac Siu', 'Ca Phe Sua', 'Cappucinno', 'Caramel Macchiato',
                'Espresso', 'Mocha', 'So-co-la Da', 'Tra Dao Cam Sa', 'Tra Den Macchiato', 'Tra Gao Rang Macchiato',
                'Tra Matcha Latte', 'Tra Oi Thanh Macchiato', 'Caramel Da Xay', 'Mocha Da Xay', 'So-co-la Da Xay',
                'Sinh To Cam Xoai', 'Sinh To Viet Quat']

        uid = str(uuid.uuid4().hex)

        if df is None:
            df = self.local_db

        phone_length = 9
        phone_number = ''.join(random.choice(numbers) for i in range(phone_length))
        phone_number = '0' + phone_number

        email = name.replace('_', '') + '@gmail.com'
        email = email.lower()

        last_order = random.choice(menu)

        row = [name, uid, phone_number, email, last_order]
        df.loc[-1] = row
        df.index = df.index + 1

        df.to_csv(self.local_db_path, index=False)
        self.local_db = pd.read_csv(self.local_db_path)

        return df.sort_index()



    def cases_handling(self, result_dict, make_underdash=True):

        if result_dict['nof_faces'] == 0:
            # Nobody in front of the camera
            self.name = None
            print('Nobody          ')
            pass

        if result_dict['nof_faces'] > 0 and result_dict['recognized'] == True and result_dict['warnings'] is None:
            self.name = result_dict['who']
            if make_underdash:
                self.name = self.name.replace(" ", "_")
            event_list = ['coming', 'leaving', 'buy coffee', 'forget stuffs', 'complain something', 'ask something']
            pseudo_event = random.sample(event_list, 1)[0]
            row = []
            if self.name is not None and result_dict['fer'] is not None:

                row.append(pseudo_event)
                row.append(result_dict['fer'])
                row.append(self.name)
                row.append(True)
                row.append(str(datetime.datetime.now()))
                row.append(find_someone_info('uid', 'name', self.name, self.local_db))

                print('Perfect case, row = ', row, len(row))

                time_stop = time.monotonic()
                print(time_stop - self.time_start)
                if time_stop - self.time_start > 5:
                    add_row(self.event_log, row)
                    self.event_log.to_csv(self.event_log_path, index=False)
                    self.event_log = pd.read_csv(self.event_log_path)
                    self.time_start = time.monotonic()
                    print('             CSV WRITTEN')

        if result_dict['nof_faces'] > 0 and result_dict['recognized'] == False and result_dict['warnings'] is None:
            # event_list = ['coming', 'leaving', 'buy coffee', 'forget stuffs', 'complain something', 'ask something']
            # pseudo_event = random.sample(event_list, 1)[0]
            # date_time = datetime.datetime.now()
            # row = []
            # if self.name is not None:
            #
            #     row.append(pseudo_event)
            #     row.append(result_dict['fer'])
            #     row.append(self.name)
            #     row.append(False)
            #     row.append(datetime.datetime.now())
            #     row.append(find_someone_info('uid', 'name', self.name, self.local_db))
            #
            # print('Uncertain: row = ', row)
            #
            #     # add_row(self.event_log, row)
            #     # self.event_log.to_csv(self.event_log_path, index=False)
            #     # self.event_log = pd.read_csv(self.event_log_path)
            pass
        if result_dict['nof_faces'] > 0 and result_dict['recognized'] == False and result_dict['warnings'] is not None:
            self.name = None
            print('Face overlapped')
            pass


pdCSV = writeToCsv()
# row = ['going', 'happy', 'Tuan_Tran', True, datetime.datetime.now(), find_someone_uid('Tuan_Tran', local_db)]
# print(find_someone_info('uid', 'name', 'Tuan_Tran', local_db))

# {'nof_faces': 0, 'recognized': False, 'who': None, 'highest_score_class': None, 'fer': None, 'warnings': None}
# Nobody

# {'nof_faces': 1, 'recognized': True, 'who': 'Tuan Tran', 'highest_score_class': None, 'fer': 'neutral', 'warnings': None}
# Perfect guess

# {'nof_faces': 1, 'recognized': False, 'who': None, 'highest_score_class': 'Tran Van Khanh', 'fer': 'sad', 'warnings': None}
# Unknown/Uncertain cases

# {'nof_faces': 1, 'recognized': False, 'who': None, 'highest_score_class': None, 'fer': None, 'warnings': 'Face Overlapped'}
# Face overlapped
