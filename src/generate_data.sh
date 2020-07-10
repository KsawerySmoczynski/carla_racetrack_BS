#!/bin/bash

for map in 'RaceTrack' 'RaceTrack2' 'circut_spa';
  do for speed in 80 110 150;
    do python runner.py -e=1 --map=$map --speed=$speed --no_agents=5;
  done;
done;

for map in 'RaceTrack' 'RaceTrack2';
  do for speed in 80 110 150;
    do python runner.py -e=1 --invert=True --map=$map --speed=$speed --no_agents=7;
  done;
done;