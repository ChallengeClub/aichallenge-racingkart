#!/bin/bash

tar \
  --exclude='*/__pycache__' \
  --exclude='*/.pytest_cache' \
  --exclude='*.pyc' \
  -zcvf submit/aichallenge_submit.tar.gz \
  -C ./aichallenge/workspace/src aichallenge_submit
