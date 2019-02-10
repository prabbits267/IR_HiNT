from math import floor
from os import listdir
from os.path import join

from MatchingModel.sources.config import *
import pandas as pd

class ProcessData:
    def __init__(self):
        self.read_folder()

    def read_folder(self):
        data_folder = listdir(ORIGINAL_DATA)
        pair_list = list()

        for flder in data_folder:
            flder_name = ORIGINAL_DATA + flder +  '/' + 'original'
            if flder != '.DS_Store':
                for file in listdir(flder_name):
                    file_name = join(flder_name, file)
                    with open(file_name, 'rt', encoding='utf-8') as file_reader:
                        pair = self.read_file(file_reader.readlines(), flder)
                        if pair != None:
                            pair_list.append(pair)

        train_len = floor(len(pair_list) * 0.95)
        train_data = pair_list[:train_len]
        test_data = pair_list[train_len:]

        self.export_csv(train_data, IR_TRAIN)
        self.export_csv(test_data, IR_TEST)

    def read_file(self, text, cluster):
        title = text[0].replace('Title: ', '')
        content = text[8:]
        content = ' '.join(content)
        if content != '' and title != '' and cluster != '':
            return content, title, cluster
        return None

    def export_csv(self, pair_list, file_name):
        df = pd.DataFrame(pair_list)
        df.to_csv(join(PATH, file_name), index=None, header=False, encoding='utf-8-sig')


def main():
    pre = ProcessData()


if __name__ == '__main__':
    main()
