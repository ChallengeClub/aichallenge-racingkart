#!/bin/bash
# E2E controller diagnosis: keep random starts and sensors, remove NPC traffic only.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec $AWSIM_DIRECTORY/AWSIM.x86_64 \
    --venue citycircuit \
    --start-mode count \
    --start-count-seconds 0 \
    --vehicles 1 \
    --npcs 0 \
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
    --camera cpu \
    --lidar cpu \
    --imu off \
    --gnss off \
    --v2x off
