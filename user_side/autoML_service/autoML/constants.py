AUTOSKLEARN1 = "AutoSklearn1"
TPOT = "TPOT"
FLAML = "FLAML"
LIGHTAUTOML = 'LightAutoML'

FRAMEWORKS = [AUTOSKLEARN1, TPOT, FLAML, LIGHTAUTOML]

AUTOML_CHOICES = [
        (AUTOSKLEARN1, AUTOSKLEARN1),
        (TPOT, TPOT),
        (FLAML, FLAML),
        (LIGHTAUTOML, LIGHTAUTOML)
    ]

AUTOSKLEARN1_PARAMETERS = {
    "time_left_for_this_task": 120,
    "per_run_time_limit": 60,
    "memory_limit": 102400,
}

TPOT_PARAMETERS = {
    "generations": 5,
    "population_size": 50,
    "max_time_mins": 1,
    "max_eval_time_mins": 1,
}

FLAML_PARAMETERS = {
    "time_budget": 60,
    "verbose": 0,
}

LIGHTAUTOML_PARAMETERS = {
    "timeout": 60,
}

FRAMEWORKS_PARAMETERS = {
    AUTOSKLEARN1: AUTOSKLEARN1_PARAMETERS, 
    TPOT: TPOT_PARAMETERS, 
    FLAML: FLAML_PARAMETERS,
    LIGHTAUTOML: LIGHTAUTOML_PARAMETERS,
}

FRAMEWORK_TIME_PARAMETERS = {
    AUTOSKLEARN1: ("time_left_for_this_task", "seconds"), 
    TPOT: ("max_time_mins", "minutes"),
    FLAML: ("time_budget", "seconds"),
    LIGHTAUTOML: ("timeout", "seconds"),
}
