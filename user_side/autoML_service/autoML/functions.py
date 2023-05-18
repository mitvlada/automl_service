from autoML_service.autoML.constants import FRAMEWORK_TIME_PARAMETERS


def set_time_parameter(time: int, framework:str):
    
    parameter, unit = FRAMEWORK_TIME_PARAMETERS[framework]

    time_budget = time
    if unit == "seconds":
        time_budget = time_budget*60

    return (parameter, time_budget)
