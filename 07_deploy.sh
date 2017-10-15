#!/bin/bash
cp optimised_model/optimised_graph.pb ./detector_graph.pb
cp ./detector_graph.pb ../carla-brain/ros/src/tl_detector/detector_graph.pb

