#!/bin/bash
set -e

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

exec "$@"

