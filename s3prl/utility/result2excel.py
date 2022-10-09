import re
import os
import argparse
import numpy as np
import pandas as pd

class Task(object):
    rules = {
        "SE": r".*(stoi|pesq) of [0-9]+ utts is ([0-9\.]+)",
    }
    default_rule = r".*(wer|acc|der|utts|MAP|EER|BLEU)[=: ]+([0-9\.]+)"

    def __init__(self, task_name, upstream, root):
        self.path = os.path.join(root, task_name, upstream)
        self.task_name = task_name
        self.upstream = upstream
        self.data = dict()

    def get_data(self):
        if "QBE" in self.task_name:
            self.get_data_qbe_()
        elif "ASV" in self.task_name:
            self.get_data_asv_()
        else:
            for pardir, dirs, files in os.walk(self.path):
                self.get_data_(pardir, files)
        return self

    def get_data_(self, pardir, files):
        for file in files:
            if file.endswith(".result"):
                split = re.findall(r'(dev|test).*', file)[0]
                fpath = os.path.join(pardir, file)
                with open(fpath, "r") as f:
                    text = f.read()
                task_name, upstream, lr = re.findall(
                    r".*/(.*)/(.*)/(.*)", pardir)[0]
                assert self.task_name == task_name, "task_name is wrong, self.task_name: {}, task_name: {}".format(
                    self.task_name, task_name)
                assert self.upstream == upstream, "upstream is wrong"
                result = re.findall(self.rules.get(
                    self.task_name, self.default_rule), text)
                result = list(
                    map(lambda x: ("{} {}".format(split, x[0]), x[0], x[1]), result))
                if lr in self.data:
                    self.data[lr] += result
                else:
                    self.data[lr] = result

    def get_data_qbe_(self):
        layer_name = os.listdir(self.path)
        result = None
        best_layer = None
        for ln in layer_name:
            with open(os.path.join(self.path, ln, "scoring", "dev.result"), 'r') as f:
                text = f.read()
            tmp = re.findall(self.default_rule, text)
            if result is None or float(tmp[0][1]) > float(result[0][1]):
                best_layer = ln
                result = tmp

        result = list(
            map(lambda x: ("dev {}".format(x[0]), x[0], x[1]), result))
        with open(os.path.join(self.path, best_layer, "scoring", "test.result"), 'r') as f:
            text = f.read()
        tmp = re.findall(self.default_rule, text)
        result += list(
            map(lambda x: ("test {}".format(x[0]), x[0], x[1]), tmp))

        self.data["lr1"] = result

        return self

    def get_data_asv_(self):
        lrs = os.listdir(self.path)
        for lr in lrs:
            self.data[lr] = []
            for split in ["dev", "test"]:
                tmp = 0
                for i in range(1, 4):
                    with open(os.path.join(self.path, lr, "seed" + str(i), "{}.result".format(split)), 'r') as f:
                        text = f.read()
                    tmp += float(re.findall(self.default_rule, text)[0][1])
                self.data[lr].append((split, "EER", str(tmp / 3)))

    @staticmethod
    def generate_dataframe():
        col_index = pd.MultiIndex.from_tuples([("tmp", "tmp1")])
        df = pd.DataFrame(index=pd.MultiIndex.from_tuples(
            [(None, None)]), columns=col_index)[1:]
        return df

    def merge_dataframe(self, df=None):
        if df is None:
            df = self.generate_dataframe()

        for lr in self.data:
            data = {}
            col_index = []
            values = []
            index = pd.MultiIndex.from_tuples([(self.upstream, lr)])
            for subrow in self.data[lr]:
                split, metrics, value = subrow
                values.append(value)
                col_index.append((self.task_name, split))
            col_index = pd.MultiIndex.from_tuples(col_index)

            def func(a, b):
                return a.where(~a.isna(), b)
            if index[0] in df.index:
                s = pd.DataFrame([values], index=index, columns=col_index)
                df = df.combine(s, func, overwrite=False)
            else:
                s = pd.Series(values, index=col_index, name=index[0])
                df = df.append(s)

        # print(df)
        return df


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='path to result', default="./result")
    parser.add_argument('-u', '--upstream', type=str, help="upstream")
    parser.add_argument('-o', '--output', type=str,
                        help="output summary path (filename will be the same as upstream)", default="./")
    return parser.parse_args()


if __name__ == "__main__":
    args = getargs()
    df = Task.generate_dataframe()
    cols = ["PR", "SID", "ER", "ASR", "QBE", "ASV", "SD", "SS", "SE", "ST"]
    for taskname in cols:
        try:
            task = Task(taskname, args.upstream, args.path).get_data()
            df = task.merge_dataframe(df)
        except TypeError as e:
            print(taskname)
            raise e
    del df["tmp"]

    tmp = {col: [] for col in cols}
    i = 0
    for col in df.columns:
        tmp[col[0]].append(i)
        i += 1
    tmp["SE"] = [-((i & 1) << 1) + 1 + j for i, j in enumerate(tmp["SE"])]
    tmp["QBE"] = [-((i & 1) << 1) + 1 + j for i, j in enumerate(tmp["QBE"])]
    index = []
    for col in cols:
        index += tmp[col]

    df.iloc[:, df.columns == ('SD', "test der")] = (df.iloc[:, df.columns == (
        'SD', "test der")].astype(np.float128) * 100).astype(str).replace("nan", np.nan)

    df.to_excel(os.path.join(args.output, "{}.xlsx".format(
        args.upstream)), header=True, startrow=1, columns=df.columns[index])
    
    ## select the best dev result ##
    cols_good = ["PR", "SID", "ER", "ASR", "QBE", "ASV", "SD", "SS", "SE", "ST"]
    for col in df.columns:
        print(df[col])
