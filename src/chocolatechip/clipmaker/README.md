run

```bash
# cd /mnt/hdd/data/video_pipeline/tracking
# find "$(pwd)" -type f -ctime -9 \( -path "*tracking/26_*" -o -path "*tracking/25_*" \)
# by the way ctime -9 means days
# find "$(pwd)" -type f -ctime -30 \( -path "*tracking/07_2022*" -o -path "*tracking/07_2022*" \)
# 
# we shouldnt need to do the above anymore
# python arglistgen.py --camera_id 07 --year 2022 --days 30
# python arglistgen.py --camera_id 24 --year 2025 --days 6
python arglistgen.py --camera_id 25 --year 2025 --days 2
python arglistgen.py --camera_id 26 --year 2025 --days 2
# ./run_script_2.sh -i 3287 -c 24
./run_script_2.sh -i 3248 -c 27
```

then cd to the folder, make sure
~/sprinkles_output is empty and then `chip sprinkles`