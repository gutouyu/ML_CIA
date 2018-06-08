import tensorflow as tf

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


# 1. Read the Census Data

# 2. Converting Data into Tensors
def input_fn(data_file, num_epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个。record_defaults用于指明每一列的缺失值用什么填充
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K') # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise

    dataset = tf.data.TextLineDataset(data_file) \
                .map(parse_csv, num_parallel_calls=5)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'] + _NUM_EXAMPLES['validation'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

# 3. Select and Engineer Features for Model

## 3.1 Base Categorical Feature Columns
# 如果我们知道所有的取值，并且取值不是很多
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'
    ]
)
# 如果不知道有多少取值
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000
)

education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
    ]
)

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed']
)


workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


# 3.2 Base Continuous Feature Columns
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

#Sometimes the relationship between a continuous feature and the label is not linear. As a hypothetical example, a person's income may grow with age in the early stage of one's career, then the growth may slow at some point, and finally the income decreases after retirement. In this scenario, using the raw age as a real-valued feature column might not be a good choice because the model can only learn one of the three cases:

# 3.2.1 连续特征离散化
# 之所以这么做是因为：有些时候连续特征和label之间不是线性的关系。可能刚开始是正的线性关系，后面又变成了负的线性关系，这样一个折线的关系整体来看就不再是线性关系。
# bucketization 装桶
# 10个边界，11个桶
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# 3.3 组合特征/交叉特征
education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)

age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
)


# 4. 模型
"""
之前的特征：
1. CategoricalColumn
2. NumericalColumn
3. BucketizedColumn
4. CrossedColumn
这些特征都是FeatureColumn的子类，可以放到一起
"""
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]

crossed_column = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000
    ),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
    )
]

model_dir = "./model/wide_component"
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_column
)

train_file = './data/adult.data'
val_file = './data/adult.data'
test_file = './data/adult.test'

# 5. Train & Evaluate & Predict
model.train(input_fn=lambda: input_fn(data_file=train_file, num_epochs=1, shuffle=True, batch_size=512))
results = model.evaluate(input_fn=lambda: input_fn(val_file, 1, False, 512))
for key in sorted(results):
    print("{0:20}: {1:.4f}".format(key, results[key]))


pred_iter = model.predict(input_fn=lambda: input_fn(test_file, 1, False, 1))
for pred in pred_iter:
    print(pred)
    break #太多了，只打印一条

test_results = model.evaluate(input_fn=lambda: input_fn(test_file, 1, False, 512))
for key in sorted(test_results):
    print("{0:20}: {1:.4f}".format(key, test_results[key]))

# 6. 正则化
model = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_column, model_dir=model_dir,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0
    )
)



# if __name__ == '__main__':
#     print(tf.VERSION)
#     data_file = './data/adult.data'
#     next_batch = input_fn(data_file, num_epochs=1, shuffle=True, batch_size=5)
#     with tf.Session() as sess:
#         first_batch = sess.run(next_batch)
#         print(first_batch[0])
#         print(first_batch[1])