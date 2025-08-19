The Intersection Configurator allows the user to see
intersection-specific parameters, such as the coordinates
of the lane markings. It also allows the user to conduct
thin plate spline rectification, producing TPS map files
to be uploaded to maltlab in the dir `/mnt/hdd/data/video_pipeline/tps`

and then the user can edit the `/mnt/hdd/pipeline/fastmot_offline/fastmot/tracker.py`
to set the TPS_Files to point to the new one. Also, set the 
`/mnt/hdd/pipeline/tracks_processing/run_DB.py` to do the same thing-- point to a new
TPS file.


The workflow is as follows. Start the GUI.

```bash
python intersection_configurator.py
```

Make sure you have a snapshot of a frame from a fisheye video,
ideally with no vehicles around (but not required) so that it is
easier to mark the distinct features such as crosswalk corners. 
Also, have a topdown Google Maps snapshot. Both pictures must be
1280x960.

The tabs "Unwrapper", "Point Pairer", and "Mapper" should be used
sequentially. First, you unwrap the raw fisheye image. Then, you use
that unwrapped image as the first photo in the Point Pairer. The second
photo is the topdown Google Maps. Click around 15-30 distinct points
to use as a reference for TPS mapping.

Once the .out file of points is exported, go to Mapper and use:
the raw fisheye snapshot, Google Maps snapshot, and .out file so that
the TPS file can be generated.
