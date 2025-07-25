Many progams are available in this directory for
facilitating use of the data pipeline.

Generally, the order of programs used is as follows.

```python
chip offline # start the pipeline for offline mp4 processing.
inventory/   # use the pipeline by moving videos.
clipmaker/   # after pipeline is done, make near miss clips.
sprinkles/   # augment the clips with spatial information and view them for filtering
unflag/      # unflag the wrongly classified near misses.
heatmap/     # produce plots of volume and conflict rates.
conflict/    # produce plots of conflict categories and more.
```

