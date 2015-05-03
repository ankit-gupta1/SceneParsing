#!/bin/sh
rm -rf *.pyc
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ../../../../opt/PYTHON/bin/python cnn.py
