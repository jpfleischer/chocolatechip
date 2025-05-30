#!/bin/bash

# TIMESTAMPS="'2024-07-26 07:00:00.000' '2024-07-26 19:00:00.000' '2024-07-27 07:00:00.000' '2024-07-27 19:00:00.000' '2024-07-28 07:00:00.000' '2024-07-28 19:00:00.000' '2024-08-05 07:00:00.000' '2024-08-05 19:00:00.000' '2024-08-06 07:00:00.000' '2024-08-06 19:00:00.000' '2024-08-07 07:00:00.000' '2024-08-07 19:00:00.000' '2024-08-08 07:00:00.000' '2024-08-08 19:00:00.000' '2024-08-26 07:00:00.000' '2024-08-26 19:00:00.000' '2024-08-27 07:00:00.000' '2024-08-27 19:00:00.000' '2024-08-28 07:00:00.000' '2024-08-28 19:00:00.000' '2024-08-29 07:00:00.000' '2024-08-29 19:00:00.000' '2024-08-30 07:00:00.000' '2024-08-30 19:00:00.000' '2024-08-31 07:00:00.000' '2024-08-31 19:00:00.000' '2024-09-01 07:00:00.000' '2024-09-01 19:00:00.000' '2024-09-02 07:00:00.000' '2024-09-02 19:00:00.000' '2024-09-05 07:00:00.000' '2024-09-05 19:00:00.000'"
# Read timestamps from arglist.txt and store them in a variable
if [[ -f "arglist.txt" ]]; then
    TIMESTAMPS=$(cat arglist.txt)
else
    echo "Error: arglist.txt not found!"
    exit 1
fi

# Default values
INTERSECTION_ID="5060"
CAMERA_ID="07"

# Parse command-line options
while getopts "i:c:" opt; do
    case $opt in
        i)
            INTERSECTION_ID="$OPTARG"
            ;;
        c)
            CAMERA_ID="$OPTARG"
            ;;
        *)
            echo "Usage: $0 [-i INTERSECTION_ID] [-c CAMERA_ID]"
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# Create necessary directories
mkdir -p $INTERSECTION_ID $INTERSECTION_ID/v2v $INTERSECTION_ID/p2v

process_3252 () {
    echo "Processing intersection $INTERSECTION_ID"
    
    

    # Run scripts with TIMESTAMPS variable
    python3 lot.py $INTERSECTION_ID $CAMERA_ID $TIMESTAMPS
    echo "Running speed_vs_TTC.py..."
    python3 speed_vs_TTC.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running TMC_Codes.py..."
    python3 TMC_Codes.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running binconflicts.py..."
    python3 binconflicts_1hagg.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running bincpi.py..."
    python3 bincpi.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running binvolume.py..."
    python3 binvolume.py $INTERSECTION_ID $CAMERA_ID $TIMESTAMPS
    echo "Running binbrake.py..."
    python3 bindeceleration.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running speedall.py..."
    python3 speedall.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running binspeed.py..."
    python3 binspeed.py $INTERSECTION_ID
    echo "Running ranking_events.py..."
    python3 ranking_events.py $INTERSECTION_ID $TIMESTAMPS
    # sed -i '1s/^/index,/g' $INTERSECTION_ID/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    # sed -i '1s/^/index,/g' $INTERSECTION_ID/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    python3 create_nearmiss_clips.py 3252/sorted_ranking.csv 3252/p2v_events.csv
}


process_5060 () {
    echo "Processing intersection $INTERSECTION_ID"
    
    # Create necessary directories
    mkdir -p $INTERSECTION_ID $INTERSECTION_ID/v2v $INTERSECTION_ID/p2v

    # Run scripts with TIMESTAMPS variable
    python3 lot.py $INTERSECTION_ID $CAMERA_ID $TIMESTAMPS
    echo "Running speed_vs_TTC.py..."
    python3 speed_vs_TTC.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running TMC_Codes.py..."
    python3 TMC_Codes.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running binconflicts.py..."
    python3 binconflicts_1hagg.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running bincpi.py..."
    python3 bincpi.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running binvolume.py..."
    python3 binvolume.py $INTERSECTION_ID $CAMERA_ID $TIMESTAMPS
    echo "Running binbrake.py..."
    python3 bindeceleration.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running speedall.py..."
    python3 speedall.py $INTERSECTION_ID $TIMESTAMPS
    echo "Running binspeed.py..."
    python3 binspeed.py $INTERSECTION_ID
    echo "Running ranking_events.py..."
    python3 ranking_events.py $INTERSECTION_ID $TIMESTAMPS
    # sed -i '1s/^/index,/g' $INTERSECTION_ID/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    # sed -i '1s/^/index,/g' $INTERSECTION_ID/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    python3 create_nearmiss_clips.py 5060/sorted_ranking.csv 5060/p2v_events.csv
}

process_3265_events () {
    echo "Processing intersection 3265"
    mkdir 3265
    #echo "Running ranking_events.py..."
    #python3 ranking_events.py 3265 '2024-04-03 07:00:00.000' '2024-04-03 19:00:00.000' '2024-04-04 07:00:00.000' '2024-04-04 19:00:00.000' '2024-04-05 07:00:00.000' '2024-04-05 19:00:00.000' '2024-04-06 07:00:00.000' '2024-04-06 19:00:00.000' '2024-04-07 07:00:00.000' '2024-04-07 19:00:00.000' '2024-04-08 07:00:00.000' '2024-04-08 19:00:00.000' '2024-04-09 07:00:00.000' '2024-04-09 19:00:00.000' '2024-04-10 07:00:00.000' '2024-04-10 19:00:00.000' '2024-04-11 07:00:00.000' '2024-04-11 19:00:00.000' '2024-04-12 07:00:00.000' '2024-04-12 19:00:00.000' '2024-04-13 07:00:00.000' '2024-04-13 19:00:00.000' '2024-04-14 07:00:00.000' '2024-04-14 19:00:00.000' '2024-04-15 07:00:00.000' '2024-04-15 19:00:00.000' '2024-04-16 07:00:00.000' '2024-04-16 19:00:00.000' '2024-04-17 07:00:00.000' '2024-04-17 19:00:00.000' '2024-04-20 07:00:00.000' '2024-04-20 19:00:00.000' '2024-04-21 07:00:00.000' '2024-04-21 19:00:00.000' '2024-04-22 07:00:00.000' '2024-04-22 19:00:00.000' '2024-04-23 07:00:00.000' '2024-04-23 19:00:00.000'
    #sed -i '1s/^/index/g' 3265/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    sed -i '1s/^/index/g' 3265/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    mkdir 3265/v2v 3265/p2v
    python3 create_nearmiss_clips.py 3265/sorted_ranking.csv 3265/p2v_events.csv
}

process_3287 () {
    echo "Processing intersection 3287"
    echo "Running binconflicts.py..."
    python3 binconflicts_1hagg.py 3287 '2024-02-26 07:00:00.000' '2024-02-26 19:00:00.000' '2024-02-27 07:00:00.000' '2024-02-27 19:00:00.000' '2024-02-28 07:00:00.000' '2024-02-28 19:00:00.000' '2024-03-07 07:00:00.000' '2024-03-07 19:00:00.000' '2024-03-08 07:00:00.000' '2024-03-08 19:00:00.000' '2024-03-09 07:00:00.000' '2024-03-09 19:00:00.000' '2024-03-10 07:00:00.000' '2024-03-10 19:00:00.000' '2024-03-12 07:00:00.000' '2024-03-12 19:00:00.000' '2024-03-13 07:00:00.000' '2024-03-13 19:00:00.000' '2024-03-14 07:00:00.000' '2024-03-14 19:00:00.000' '2024-03-15 07:00:00.000' '2024-03-15 19:00:00.000' '2024-03-16 07:00:00.000' '2024-03-16 19:00:00.000'
    echo "Running left opposing through..."
    python3 lot.py 3287 24 '2024-02-26 07:00:00.000' '2024-02-26 19:00:00.000' '2024-02-27 07:00:00.000' '2024-02-27 19:00:00.000' '2024-02-28 07:00:00.000' '2024-02-28 19:00:00.000' '2024-03-07 07:00:00.000' '2024-03-07 19:00:00.000' '2024-03-08 07:00:00.000' '2024-03-08 19:00:00.000' '2024-03-09 07:00:00.000' '2024-03-09 19:00:00.000' '2024-03-10 07:00:00.000' '2024-03-10 19:00:00.000' '2024-03-12 07:00:00.000' '2024-03-12 19:00:00.000' '2024-03-13 07:00:00.000' '2024-03-13 19:00:00.000' '2024-03-14 07:00:00.000' '2024-03-14 19:00:00.000' '2024-03-15 07:00:00.000' '2024-03-15 19:00:00.000' '2024-03-16 07:00:00.000' '2024-03-16 19:00:00.000'
    echo "Running speed_vs_TTC.py..."
    python3 speed_vs_TTC.py 3287 '2023-08-23 07:00:00.000' '2023-08-23 19:00:00.000' '2023-08-24 07:00:00.000' '2023-08-24 19:00:00.000'
    echo "Running bincpi.py..."
    python3 bincpi.py 3287 '2023-08-23 07:00:00.000' '2023-08-23 19:00:00.000' '2023-08-24 07:00:00.000' '2023-08-24 19:00:00.000'
    echo "Running binvolume.py..."
    python3 binvolume.py 3287 24 '2023-08-23 07:00:00.000' '2023-08-23 19:00:00.000' '2023-08-24 07:00:00.000' '2023-08-24 19:00:00.000'
    echo "Running binbrake.py..."
    python3 bindeceleration.py 3287 '2023-08-23 07:00:00.000' '2023-08-23 19:00:00.000' '2023-08-24 07:00:00.000' '2023-08-24 19:00:00.000'
    echo "Running speedall.py..."
    python3 speedall.py 3287 '2023-08-23 07:00:00.000' '2023-08-23 19:00:00.000' '2023-08-24 07:00:00.000' '2023-08-24 19:00:00.000'
    echo "Running binspeed.py..."
    python3 binspeed.py 3287
}

process_3287_events () {
    echo "Processing intersection 3287"
    echo "Running ranking_events.py..."
    python3 ranking_events.py $INTERSECTION_ID $TIMESTAMPS
    # sed -i '1s/^/index/g' 3287/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    # sed -i '1s/^/index/g' 3287/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    mkdir 3287/v2v 3287/p2v
    python3 create_nearmiss_clips.py 3287/sorted_ranking.csv 3287/p2v/p2v_events.csv
}

process_3248_events () {
    echo "Processing intersection 3248"
    mkdir -p 3248
    #echo "Running ranking_events.py..."
    # python3 ranking_events.py 3248 '2024-10-29 07:00:00.000' '2024-10-29 19:00:00.000' '2024-10-30 07:00:00.000' '2024-10-30 19:00:00.000' '2024-10-31 07:00:00.000' '2024-10-31 19:00:00.000' '2024-11-01 07:00:00.000' '2024-11-01 19:00:00.000' '2024-11-02 07:00:00.000' '2024-11-02 19:00:00.000' '2024-11-03 07:00:00.000' '2024-11-03 19:00:00.000' '2024-11-06 07:00:00.000' '2024-11-06 19:00:00.000'
    python3 ranking_events.py $INTERSECTION_ID $TIMESTAMPS
    sed -i '1s/^/index/g' 3248/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    sed -i '1s/^/index/g' 3248/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    mkdir -p 3248/v2v 3248/p2v
    python3 create_nearmiss_clips.py 3248/sorted_ranking.csv 3248/p2v/p2v_events.csv
}

process_3032_events () {
    echo "Processing intersection 3032"
    mkdir 3032
    #echo "Running ranking_events.py..."
    #python3 ranking_events.py 3032 '2024-04-03 07:00:00.000' '2024-04-03 19:00:00.000' '2024-04-04 07:00:00.000' '2024-04-04 19:00:00.000' '2024-04-05 07:00:00.000' '2024-04-05 19:00:00.000' '2024-04-06 07:00:00.000' '2024-04-06 19:00:00.000' '2024-04-07 07:00:00.000' '2024-04-07 19:00:00.000' '2024-04-08 07:00:00.000' '2024-04-08 19:00:00.000' '2024-04-10 07:00:00.000' '2024-04-10 19:00:00.000' '2024-04-11 07:00:00.000' '2024-04-11 19:00:00.000' '2024-04-13 07:00:00.000' '2024-04-13 19:00:00.000' '2024-04-14 07:00:00.000' '2024-04-14 19:00:00.000' '2024-04-15 07:00:00.000' '2024-04-15 19:00:00.000' '2024-04-16 07:00:00.000' '2024-04-16 19:00:00.000' '2024-04-17 07:00:00.000' '2024-04-17 19:00:00.000' '2024-04-19 07:00:00.000' '2024-04-19 19:00:00.000' '2024-04-20 07:00:00.000' '2024-04-20 19:00:00.000' '2024-04-21 07:00:00.000' '2024-04-21 19:00:00.000' '2024-04-22 07:00:00.000' '2024-04-22 19:00:00.000' '2024-04-23 07:00:00.000' '2024-04-23 19:00:00.000'
    #sed -i '1s/^/index/g' 3032/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    sed -i '1s/^/index/g' 3032/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    mkdir 3032/v2v 3032/p2v
    python3 create_nearmiss_clips.py 3032/sorted_ranking.csv 3032/p2v/p2v_events.csv
}

process_3334_events () {
    echo "Processing intersection 3334"
    mkdir 3334
    echo "Running ranking_events.py..."
    python3 ranking_events.py $INTERSECTION_ID $TIMESTAMPS
    sed -i '1s/^/index/g' 3334/sorted_ranking.csv
    echo "Running p2vevents.py..."
    python3 p2v_events.py $INTERSECTION_ID $TIMESTAMPS
    sed -i '1s/^/index/g' 3334/p2v_events.csv
    echo "Running create_nearmiss_clips.py..."
    mkdir 3334/v2v 3334/p2v
    python3 create_nearmiss_clips.py 3334/sorted_ranking.csv 3334/p2v/p2v_events.csv
}

# Check if the 'ts' command exists
if ! command -v ts &> /dev/null
then
    echo "Error: 'ts' command not found. Please install 'moreutils' package. sudo apt-get install moreutils"
    exit 1
fi

# Choose which process function to run based on the provided INTERSECTION_ID
case "$INTERSECTION_ID" in
    "3252")
        process_3252
        ;;
    "5060")
        process_5060
        ;;
    "3265")
        process_3265_events
        ;;
    "3287")
        # For intersection 3287, if you have two modes (process_3287 vs process_3287_events),
        # you can prompt the user to choose one:
        echo "Intersection 3287 has multiple processing modes."
        read -p "Enter 1 for process_3287 or 2 for process_3287_events: " mode
        if [ "$mode" = "1" ]; then
            process_3287
        else
            process_3287_events
        fi
        ;;
    "3248")
        process_3248_events
        ;;
    "3032")
        process_3032_events
        ;;
    "3334")
        process_3334_events
        ;;
    *)
        echo "Unknown intersection: $INTERSECTION_ID"
        exit 1
        ;;
esac

