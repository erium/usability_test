import pandas as pd
from joblib import dump, load

model_data = load('./out/model.joblib')
model_name = model_data[0]
model_params = model_data[1][2]
labels = model_data[2]
pca = model_data[1][3]
model = model_data[1][4]
target = model_data[3]

scaler_x = load('./out/scaler_x.joblib')
scaler_y = load('./out/scaler_y.joblib')


def predict(input_train_df, target):

    input_labels = input_train_df.columns
    inputs = {}
    for inp in input_labels:
        minval = input_train_df[inp].min()
        maxval = input_train_df[inp].max()
        inputs[inp] = input(inp + " (between {} and {})".format(minval, maxval))

    inputs = pd.DataFrame([inputs])
    inputs = scaler_x.transform(inputs)
    inputs = pd.DataFrame(data=inputs, columns=input_labels)
    inputs = pca.transform(inputs)
    if model_name == 'poly':
        poly_transform = PolynomialFeatures(degree=model_params["model__degree"])
        inputs = poly_transform.fit_transform(inputs)

    outputs = pd.DataFrame(model.predict(inputs))
    outputs = pd.DataFrame(scaler_y.inverse_transform(outputs))
    outputs.columns = ['Predicted ' + x for x in target]

    return outputs