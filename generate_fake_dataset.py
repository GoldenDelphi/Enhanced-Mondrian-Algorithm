import random

def dataset():
    columns = ['age', 'gender', 'zipcode', 'salary']
    feature_columns = ['age', 'gender', 'zipcode']
    sensitive_column = 'salary'
    random.seed(10)
    data = [[random.randint(18, 45), random.choice(['M', 'F']),random.randint(500000, 600000), int(random.randint(1, 300) * 1e5)] for _ in range(5000)]
    return columns, feature_columns, sensitive_column, data