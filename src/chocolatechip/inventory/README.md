This program should be used to run videos through the pipeline.

python video_pipeline_checker.py --dryrun

this will tell you which videos are missing and havent been analyzed/ingested
into the pipeline yet. it also cuts up videos into 15 minute segments

```bash
# after changing to the intersection you want and the day of week
python video_pipeline_checker.py
ssh maltlab
# all files modified (or created) in the last 3 days
find /mnt/hdd/data/video_pipeline -type f -ctime -3 -name "*.mp4"
```

