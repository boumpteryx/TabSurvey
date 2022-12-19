from comet_ml import Experiment
import time

def init_comet(args, project_name="train_tab", api_key=""):
    if project_name=="" or project_name==None:
        return None

    timestamp = time.time()
    args["timestamp"] = timestamp
    workspace = args.get("workspace", "yamizi")
    xp = args.get("model_name", "")
    api_key = args.get("api_key", api_key)
    experiment_name = "{}_{}_{}".format(xp, args.get("dataset",""), timestamp)
    experiment = Experiment(api_key=api_key,
                            project_name=project_name,
                            workspace=workspace,
                            auto_param_logging=False, auto_metric_logging=False,
                            parse_args=False, display_summary=False, disabled=False)

    experiment.set_name(experiment_name)
    experiment.log_parameters(args)

    return experiment