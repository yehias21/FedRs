#!/usr/bin/env bash

export PYTHONPATH=.

tmux kill-server

tmux new-session -d -s "s"
tmux send-keys -t "s" "conda activate flower-secagg;echo 'Starting Server !';python src/core/servers/server.py --sim 0" Enter

tmux new-session -d -s "c1"
tmux send-keys -t "c1" "conda activate py39;python src/core/clients/client.py --cid 1" Enter

tmux new-session -d -s "c2"
tmux send-keys -t "c2" "conda activate py39;python src/core/clients/client.py --cid 2" Enter

tmux join-pane -s c2 -t c1 -h
#tmux join-pane -s c1 -t s
tmux attach -t c1