#!/bin/bash
# TinyLiDARNet teacher collection: E2E-like traffic/random start with privileged
# localization sensors enabled only for the MPC teacher.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec $AWSIM_DIRECTORY/AWSIM.x86_64 \
    --venue citycircuit \
    --start-mode count \
    --start-count-seconds 0 \
    --vehicles 1 \
    --npcs 2 \
    --boosts 2 \
    --laps 6 \
    --timeout 10000000.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --start-random on \
    --ranking off \
    --camera off \
    --lidar cpu \
    --imu on \
    --gnss on \
    --v2x off
