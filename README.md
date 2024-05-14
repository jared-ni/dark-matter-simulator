### Instructions
1. Set the initial state of the universe in `initial_state.txt` (see next section for format).
    - You can generate a random initial state by running `python3 input.py > initial_state.txt`. Set the number of objects within `input.py` (the variable is called `n`).
2. Set the desired parameters at the top of `gravity.cpp`.
    - Set all to `false` to accurately evaluate runtime, which is printed in the `.err` file.
    - Set `OUTPUT_VIDEO_FRAMES` to `true` if you want to see the video (see Step 4).
    - Set `CHECK_MOMENTUM` to `true` to calculate momentum and print it in `.err`.
    - Set `BENCHMARK_PAPI` to `true` to use PAPI, and set `PAPI_TYPE_DESCRIPTION`. The result is printed in the `.err` file.
3. Run using `sbatch job.sh`.
    - You can change the number of nodes and ranks there.
4. If `OUTPUT_VIDEO_FRAMES` is `true`, the `.out` file will contain pixel data corresponding to video frames. Use `ffmpeg` to generate the video.
    - Download the `.out` file.
    - Run `cat <name of .out file> | ffmpeg -y -f rawvideo -pixel_format gbrp -video_size 1024x768 -i - -c:v h264 -pix_fmt yuv420p video.mov`.
    - Open `video.mov` with a video player.
    - The version of `ffmpeg` in the repository is for Windows. Visit [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) for other versions.
    - You may need to add `.../ffmpeg/bin/` to the `PATH` environment variable.

### Format for `initial_state.txt`
Syntax:
- First line: `num_time_steps` `num_objects`
- Next `num_objects` lines: `mass` `radius` `pos.x` `pos.y` `pos.z` `vel.x` `vel.y` `vel.z`

Meaning:
- `num_time_steps`: Number of time steps for which to run the simulation
- `num_objects`: Number of objects in the initial state of the universe
- Each of the lines after the first line gives the initial state of one object
- `mass`: Mass of object
- `radius`: Radius of object
- `pos.x`, `pos.y`, `pos.z`: x-, y-, and z-components of the object's initial position
- `vel.x`, `vel.y`, `vel.z`: x-, y-, and z-components of the object's initial velocity
