import pandas as pd
import gc

IGNORE_COLS = [
    'id'
]
CATE_COLS = [
    'feat_cat_1', 'feat_cat_2'
]
NUMERIC_COLS = [
    'feat_num_1', 'feat_num_2'
]
data_path = './data/train.csv'


def convert_to_ffm_file(df, label=None):
    """
    把pandas.DataFrame转换成libffm格式文件

    :param df: pd.DataFrame
    :param label: string，表示df中label是哪一列
    :return: 转换之后的libffm格式的文件
    """
    assert label is not None

    y = df[label].values.tolist()
    df.drop([label], axis=1, inplace=True)

    # 1. 编号Field Feature
    # 对field编码
    field_dict = dict(zip(df.columns, range(0, len(df.columns))))

    # 对feature编码
    feature_dict = dict()
    feature_cnt = 0
    for col in df.columns:
        if col in IGNORE_COLS:
            continue
        if col == label:
            continue

        unique_vals = df[col].unique()
        if col in NUMERIC_COLS:
            # 编号连续特征
            feature_dict[col] = feature_cnt
            feature_cnt += 1
        else:
            # 编号类别特征
            feature_dict[col] = dict(zip(unique_vals, range(feature_cnt, feature_cnt + len(unique_vals))))
            feature_cnt += len(unique_vals)

    # 2. 转换libffm_file
    # dest_file = './data/ta.libffm'
    # for index, row in df.iterrows():
    #     line = line + row[label] + " "
    #     # row是Series
    #     for key in row.index:
    #         if key == label:
    #             # TODO






train = pd.read_csv(data_path)
cols = [c for c in train.columns if c not in IGNORE_COLS]

train_ffm = convert_to_ffm_file(train[cols], label='target')











