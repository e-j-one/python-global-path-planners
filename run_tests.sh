#!/usr/bin/env bash

echo "Starting unit tests for global path planner."

export PYTHONPATH=$(pwd)

pytest --cov-report term-missing --cov=path_planners/
# pytest --cov=path_planners/

echo "Test Ended."