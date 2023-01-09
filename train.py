import logging
import sys

import optuna
import torch
from optuna.trial import TrialState
from optuna._callbacks import MaxTrialsCallback, RetryFailedTrialCallback
from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split

from utils.comet import init_comet


def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)


        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save all the statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)
    # f = open("output_final/model_" + args.dataset + "_" + args.model_name + "_params.json", "w")
    # f.write(str(curr_model.params))
    # f.close()
    # torch.save(curr_model, "output_final/model_" + args.dataset + "_" + args.model_name + "_final.pt")

    return sc, (train_timer.get_average_time(), test_timer.get_average_time())


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        args = {**vars(self.args), **trial_params}

        experiment = init_comet(args=args, project_name="tabsurvey_train")
        model = self.model_name(trial_params, self.args, experiment)


        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)
        experiment.log_metrics(sc.get_results())
        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    n_to_finish = args.n_trials
    print(f"{n_completed} already completed")
    if n_completed < n_to_finish:
        study.optimize(
            Objective(args, model_name, X, y), 
            n_trials=None,
            callbacks=[
                MaxTrialsCallback(
                    n_to_finish,
                    states=(TrialState.COMPLETE,),
                ),        
            ],
        )
    print("Best parameters:", study.best_trial.params)

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)
    cross_validation(model, X, y, args, save_model=True)


def main_once(args):
    print("Train model with given hyperparameters")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    sc, time = cross_validation(model, X, y, args)
    print(sc.get_results())
    print(time)

all_models = ["CatBoost"] # "LinearModel", "DeepFM", "RLN", , "LinearModel", "KNN", "DecisionTree", "RandomForest", "XGBoost", "LightGBM", "ModelTree",
               # "MLP", "TabNet", "VIME", ,"DeepGBM", "STG", "NAM", "DANet", "NODE", "DNFNet", "CatBoost"
#                "SAINT",  "VIME", "TabTransformer"

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

    if arguments.optimize_hyperparameters:
        if len(all_models) > 1:
            for model in all_models:
                arguments.model_name = model
                print("running ", arguments.model_name)
                print("on ", arguments.dataset)
                main(arguments)
        else:
            print("running ", arguments.model_name)
            print("on ", arguments.dataset)
            main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        main_once(arguments)
