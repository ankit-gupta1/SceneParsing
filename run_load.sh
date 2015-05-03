#!/bin/sh
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ../../../../opt/PYTHON/bin/python load_data.py
