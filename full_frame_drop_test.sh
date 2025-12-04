LD_LIBRARY_PATH=libs:${LD_LIBRARY_PATH} python3 grace_entropy_packet_error.py \
    --video videos/testvideos/video-0.mp4 \
    --model_path models/grace/4096_freeze.model \
    --nframes 16 \
    --bit_error_rate 1e-5 \
    --poisson_scale 10.0 \
    --chunk_bytes 1400