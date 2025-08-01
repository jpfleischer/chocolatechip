> [!IMPORTANT]
> Conflict unique IDs are prefixed with 1
> if the unique IDs are coming from a composite camera ID.
> This is accounted for in the programs but be careful to
> remember this when writing new code.


`weeklyp2vtype.py` takes the number of days from
times_config and divides by 7 to get the number
of weeks.  it also creates plots such as:

<img width="2070" height="1470" alt="image" src="https://github.com/user-attachments/assets/576ab94a-9170-4118-a1ba-b2f683395e50" />

<img width="2370" height="1169" alt="image" src="https://github.com/user-attachments/assets/d91a2ae4-17f5-4d8e-9c48-1aff7570b41e" />



`intersection_conflict_overview.py` creates a table
with these headers:

<img width="772" height="60" alt="image" src="https://github.com/user-attachments/assets/40bec629-8069-4083-acf2-5dcbb6d6e191" />


`histogram_conflict.py` creates histogram heatmaps overlayed
on intersection pictures

<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/ae5cf5ca-5de2-47fd-acfe-ddf88e5738a4" />
