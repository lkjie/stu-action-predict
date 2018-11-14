#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import tensorflow as tf
import contextlib
import datetime
import json
import multiprocessing
import numbers
import os
import threading
import uuid
from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python import pywrap_tensorflow

_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"


class BaseBenchmarkLogger(object):
    """Class to log the benchmark information to STDOUT."""

    def log_evaluation_result(self, eval_results):
        """Log the evaluation result.
    
        The evaluate result is a dictionary that contains metrics defined in
        model_fn. It also contains a entry for global_step which contains the value
        of the global step when evaluation was performed.
    
        Args:
          eval_results: dict, the result of evaluate.
        """
        if not isinstance(eval_results, dict):
            tf.logging.warning("eval_results should be dictionary for logging. "
                               "Got %s", type(eval_results))
            return
        global_step = eval_results[tf.GraphKeys.GLOBAL_STEP]
        for key in sorted(eval_results):
            if key != tf.GraphKeys.GLOBAL_STEP:
                self.log_metric(key, eval_results[key], global_step=global_step)

    def log_metric(self, name, value, unit=None, global_step=None, extras=None):
        """Log the benchmark metric information to local file.
    
        Currently the logging is done in a synchronized way. This should be updated
        to log asynchronously.
    
        Args:
          name: string, the name of the metric to log.
          value: number, the value of the metric. The value will not be logged if it
            is not a number type.
          unit: string, the unit of the metric, E.g "image per second".
          global_step: int, the global_step when the metric is logged.
          extras: map of string:string, the extra information about the metric.
        """
        metric = _process_metric_to_json(name, value, unit, global_step, extras)
        if metric:
            tf.logging.info("Benchmark metric: %s", metric)

    def log_run_info(self, model_name, dataset_name, run_params, test_id=None):
        tf.logging.info("Benchmark run: %s",
                        _gather_run_info(model_name, dataset_name, run_params,
                                         test_id))

    def on_finish(self, status):
        pass


def _process_metric_to_json(
        name, value, unit=None, global_step=None, extras=None):
    """Validate the metric data and generate JSON for insert."""
    if not isinstance(value, numbers.Number):
        tf.logging.warning(
            "Metric value to log should be a number. Got %s", type(value))
        return None

    extras = _convert_to_json_dict(extras)
    return {
        "name": name,
        "value": float(value),
        "unit": unit,
        "global_step": global_step,
        "timestamp": datetime.datetime.utcnow().strftime(
            _DATE_TIME_FORMAT_PATTERN),
        "extras": extras}


def _collect_tensorflow_info(run_info):
    run_info["tensorflow_version"] = {
        "version": tf.VERSION, "git_hash": tf.GIT_VERSION}


def _collect_run_params(run_info, run_params):
    """Log the parameter information for the benchmark run."""

    def process_param(name, value):
        type_check = {
            str: {"name": name, "string_value": value},
            int: {"name": name, "long_value": value},
            bool: {"name": name, "bool_value": str(value)},
            float: {"name": name, "float_value": value},
        }
        return type_check.get(type(value),
                              {"name": name, "string_value": str(value)})

    if run_params:
        run_info["run_parameters"] = [
            process_param(k, v) for k, v in sorted(run_params.items())]


def _collect_tensorflow_environment_variables(run_info):
    run_info["tensorflow_environment_variables"] = [
        {"name": k, "value": v}
        for k, v in sorted(os.environ.items()) if k.startswith("TF_")]


# The following code is mirrored from tensorflow/tools/test/system_info_lib
# which is not exposed for import.
def _collect_cpu_info(run_info):
    """Collect the CPU information for the local environment."""
    cpu_info = {}

    cpu_info["num_cores"] = multiprocessing.cpu_count()

    try:
        # Note: cpuinfo is not installed in the TensorFlow OSS tree.
        # It is installable via pip.
        import cpuinfo  # pylint: disable=g-import-not-at-top

        info = cpuinfo.get_cpu_info()
        cpu_info["cpu_info"] = info["brand"]
        cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6

        run_info["machine_config"]["cpu_info"] = cpu_info
    except ImportError:
        tf.logging.warn("'cpuinfo' not imported. CPU info will not be logged.")


def _collect_gpu_info(run_info, session_config=None):
    """Collect local GPU information by TF device library."""
    gpu_info = {}
    local_device_protos = list_local_devices(session_config)

    gpu_info["count"] = len([d for d in local_device_protos
                             if d.device_type == "GPU"])
    # The device description usually is a JSON string, which contains the GPU
    # model info, eg:
    # "device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0"
    for d in local_device_protos:
        if d.device_type == "GPU":
            gpu_info["model"] = _parse_gpu_model(d.physical_device_desc)
            # Assume all the GPU connected are same model
            break
    run_info["machine_config"]["gpu_info"] = gpu_info


def _collect_memory_info(run_info):
    try:
        # Note: psutil is not installed in the TensorFlow OSS tree.
        # It is installable via pip.
        import psutil  # pylint: disable=g-import-not-at-top
        vmem = psutil.virtual_memory()
        run_info["machine_config"]["memory_total"] = vmem.total
        run_info["machine_config"]["memory_available"] = vmem.available
    except ImportError:
        tf.logging.warn("'psutil' not imported. Memory info will not be logged.")


def _parse_gpu_model(physical_device_desc):
    # Assume all the GPU connected are same model
    for kv in physical_device_desc.split(","):
        k, _, v = kv.partition(":")
        if k.strip() == "name":
            return v.strip()
    return None


def _convert_to_json_dict(input_dict):
    if input_dict:
        return [{"name": k, "value": v} for k, v in sorted(input_dict.items())]
    else:
        return []


def _gather_run_info(model_name, dataset_name, run_params, test_id):
    """Collect the benchmark run information for the local environment."""
    run_info = {
        "model_name": model_name,
        "dataset": {"name": dataset_name},
        "machine_config": {},
        "test_id": test_id,
        "run_date": datetime.datetime.utcnow().strftime(
            _DATE_TIME_FORMAT_PATTERN)}
    session_config = None
    if "session_config" in run_params:
        session_config = run_params["session_config"]
    _collect_tensorflow_info(run_info)
    _collect_tensorflow_environment_variables(run_info)
    _collect_run_params(run_info, run_params)
    _collect_cpu_info(run_info)
    _collect_gpu_info(run_info, session_config)
    _collect_memory_info(run_info)
    return run_info


def list_local_devices(session_config=None):
    """List the available devices available in the local process.
  
    Args:
      session_config: a session config proto or None to use the default config.
  
    Returns:
      A list of `DeviceAttribute` protocol buffers.
    """

    def _convert(pb_str):
        m = device_attributes_pb2.DeviceAttributes()
        m.ParseFromString(pb_str)
        return m

    return [
        _convert(s)
        for s in pywrap_tensorflow.list_devices(session_config=session_config)
    ]
