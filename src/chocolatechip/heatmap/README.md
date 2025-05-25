the temporary is just for
turning movements quick fix.

To use heatmap.py properly it really
depends on the times_config.py.

The miovision-gridsmart/video-time-parser.py
will yield the appropriate timestamps to put
into that times_config.py

Heatmaps/lineplots are generated based on the 
video files' timeframes, as some of them may be
cut off due to networking or other issues.
So, we must take into consideration the timeframes
and interpolate the rest of the time periods which
is done in heatmap.py.

