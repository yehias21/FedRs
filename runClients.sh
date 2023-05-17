#!/usr/bin/env bash
# Bash script to run server and 100 client
# Setup the environment
export PYTHONPATH=.

## Start the clients
for i in {1..20}
do
    python src/core/clients/client.py --cid "$i" &
done

## Wait for all clients to finish
wait
