import os
from tvm import autotvm
from tvm import auto_scheduler


def android_autotvm_tune_task(task, log_dir):
    from config.android_config import autotvm_tuning_option_rpc
    trials = autotvm_tuning_option_rpc["n_trial"]
    early_stopping = autotvm_tuning_option_rpc["early_stopping"]
    measure_option = autotvm_tuning_option_rpc["measure_option"]
    Tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
    log_dir_tmp = log_dir + '.tmp'
    if os.path.exists(log_dir_tmp):
        os.remove(log_dir_tmp)

    print(task.config_space)
    Tuner.tune(
        n_trial=trials, 
        early_stopping=early_stopping, 
        measure_option=measure_option, 
        callbacks=[autotvm.callback.progress_bar(trials, prefix='\n'), 
                    autotvm.callback.log_to_file(log_dir_tmp)]
    )
    autotvm.record.pick_best(log_dir_tmp, log_dir)


def android_ansor_tune_task(task, log_dir):
    from config.android_config import ansor_tuning_option_rpc, rpc_key, tracker_host, tracker_port
    trials = ansor_tuning_option_rpc["n_trial"]
    builder = ansor_tuning_option_rpc["builder"]
    runner = auto_scheduler.RPCRunner(key=rpc_key, host=tracker_host, port=tracker_port,
        repeat = ansor_tuning_option_rpc["repeat"],
        timeout = ansor_tuning_option_rpc["timeout"],
        min_repeat_ms = ansor_tuning_option_rpc["min_repeat_ms"],
        enable_cpu_cache_flush = ansor_tuning_option_rpc["enable_cpu_cache_flush"],
    )
    
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  
        builder=builder,
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_dir)],
    )
    task.tune(tune_option)

def linux_autotvm_tune_task(task, log_dir):
    from config.linux_config import autotvm_tuning_option
    trials = autotvm_tuning_option["n_trial"]
    early_stopping = autotvm_tuning_option["early_stopping"]
    measure_option = autotvm_tuning_option["measure_option"]
    Tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
    log_dir_tmp = log_dir + '.tmp'
    if os.path.exists(log_dir_tmp):
        os.remove(log_dir_tmp)

    print(task.config_space)
    Tuner.tune(
        n_trial=trials, 
        early_stopping=early_stopping, 
        measure_option=measure_option, 
        callbacks=[autotvm.callback.progress_bar(trials, prefix='\n'), 
                    autotvm.callback.log_to_file(log_dir_tmp)]
    )
    autotvm.record.pick_best(log_dir_tmp, log_dir)

def linux_ansor_tune_task(task, log_dir):
    from config.linux_config import ansor_tuning_option
    trials = ansor_tuning_option["n_trial"]
    builder = ansor_tuning_option["builder"]
    runner = auto_scheduler.LocalRunner(
        repeat = ansor_tuning_option["repeat"],
        timeout = ansor_tuning_option["timeout"],
        min_repeat_ms = ansor_tuning_option["min_repeat_ms"],
        enable_cpu_cache_flush = ansor_tuning_option["enable_cpu_cache_flush"],
    )
    
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,  
        builder=builder,
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_dir)],
    )
    task.tune(tune_option)