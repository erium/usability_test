normalisation_method = 'standard'

X = df.drop(columns=target)
y = df[target]

labels = list(X.columns)
num_labels = len(labels)
target_labels = list(y.columns)
num_target_labels = len(target_labels)
print("prediction inputs:")
print(labels)
print("\nprediciton outputs:")
print(target_labels)
print()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)


if variables_to_remove:
    X_train = X_train.drop(columns=variables_to_remove)
    X_test = X_test.drop(columns=variables_to_remove)
    
    for x in variables_to_remove:
        labels.remove(x)

path = 'out'
isExist = os.path.exists(path)
if isExist:
  for root, dirs, files in os.walk(path):
      for f in files:
          os.unlink(os.path.join(root, f))
      for d in dirs:
          shutil.rmtree(os.path.join(root, d))
else:
  os.makedirs(path)

from functions.scale import scale

if normalisation_method:
    X_train, X_test, y_train, y_test = scale(X_train, X_test, y_train, y_test, normalisation_method, path)


models = {
    'linear': LinearRegression(),
    'l1_linear': Lasso(),
    'l2_linear': Ridge(),
    'poly': Ridge(),
    'tree': DecisionTreeRegressor(max_depth=max_depth),
    'forest': RandomForestRegressor(max_depth=max_depth),
    'mlp': MLPRegressor()
}


# Parameter grids for each model
linear_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__fit_intercept": [True, False]
}

lasso_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__alpha": alpha_param,
    "model__fit_intercept": [True, False]
}

ridge_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__alpha": alpha_param,
    "model__fit_intercept": [True, False]
}

poly_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__alpha": alpha_param,
    "model__fit_intercept": [True, False],
    "poly__degree": np.arange(2, poly_limit + 1, 1),
    "poly__include_bias": [True, False]
}

tree_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__criterion": ['squared_error', 'friedman_mse', 'absolute_error'],
    "model__ccp_alpha": alpha_param
}

forest_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__criterion": ['squared_error', 'absolute_error'],
    "model__ccp_alpha": alpha_param,
    "model__n_estimators": np.linspace(1, 1000, 21, dtype=int)
}

mlp_param_grid = {
    "pca__n_components": np.linspace(1, len(labels), min(len(labels), 10), dtype=int),
    "model__activation": ['identity', 'logistic', 'tanh', 'relu'],
    "model__solver": ['lbfgs', 'sgd', 'adam'],
    "model__alpha": alpha_param,
    "model__learning_rate": ['constant', 'invscaling', 'adaptive']
}


models_param_grid = {
    'linear': linear_param_grid,
    'l1_linear': lasso_param_grid,
    'l2_linear': ridge_param_grid,
    'poly': poly_param_grid,
    'tree': tree_param_grid,
    'forest': forest_param_grid,
    'mlp': mlp_param_grid
}


from functions.run_regression_pipeline import run_regression

print("Models to train:", run_models, "\n\n")

model_results = run_regression(X_train, X_test, y_train, y_test, run_models, models, models_param_grid, labels)


def show_results():

    results = pd.DataFrame(model_results)
    results = results[:2]
    results.index = ['Testing Score', 'Time Taken']
    display(results[:2])


    plt.figure()
    plt.title("Testing Score")
    model = list(model_results.keys())
    final_time = [x[0] for x in list(model_results.values())]
    plt.bar(model, final_time)
    plt.show()


    plt.figure()
    plt.title("Training Time")
    model = list(model_results.keys())
    final_time = [x[1] for x in list(model_results.values())]
    plt.bar(model, final_time)
    plt.show()


    best_model = max(model_results, key=(lambda key: model_results[key]))
    print("Best model:", best_model)
    print("Model parameters:", model_results[best_model][2])

    from joblib import dump, load

    dump([best_model, model_results[best_model], labels, target], path + '/model.joblib')
