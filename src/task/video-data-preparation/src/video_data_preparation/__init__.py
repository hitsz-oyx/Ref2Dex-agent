"""EgoDex preprocessing; imports do not load models or initialize CUDA."""
SCHEMA = "ref2dex.egodex-shared-tracks.v1"
FPS, HISTORY, HORIZON, POINTS = 30, 4, 24, 32
SOURCE_ID = 3  # OakInk2=0, GRAB=1, ARCTIC=2 in Shared-WAM.
