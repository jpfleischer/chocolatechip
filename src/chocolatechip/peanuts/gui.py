import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
import pandas as pd
from yaspin import yaspin
from yaspin.spinners import Spinners
from chocolatechip.MySQLConnector import MySQLConnector
from datetime import datetime, timedelta

class CalGUI:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        self.root.title("Calendar")
        self.root.geometry("600x600")  # Increased size for better visibility
        self.root.resizable(False, False)

        # Calendar widget
        self.cal = Calendar(self.root, selectmode="day", date_pattern='y-mm-dd')
        self.cal.pack(pady=20)

        # Button to fetch the details for the selected day
        self.button = ttk.Button(self.root, text="See Insights", command=self.show_day_info)
        self.button.pack(pady=20)

        # Define color bins and create tags
        self.define_color_tags()

        # Apply colors to the calendar based on missing percentage
        self.apply_colors_to_calendar()

    def define_color_tags(self):
        """Define tags for different missing percentage ranges."""
        # Define bins (0-10%, 10-20%, ..., 90-100%)
        self.bins = [i for i in range(0, 101, 10)]  # 0,10,20,...,100
        self.color_tags = {}

        for i in range(len(self.bins) - 1):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            tag_name = f"color_{lower}_{upper}"
            # Calculate the midpoint percentage for color calculation
            percentage_mid = (lower + upper) / 2
            color = self.calculate_color(percentage_mid)

            # Configure the tag with the calculated color
            self.cal.tag_config(tag_name, background=color, foreground='black')  # Foreground for text readability
            self.color_tags[(lower, upper)] = tag_name

    def calculate_color(self, percentage_missing):
        """
        Convert percentage missing to a color on a gradient from green to red.
        0% -> green (#00ff00)
        100% -> red (#ff0000)
        """
        red = int(255 * (percentage_missing / 100))
        green = int(255 * ((100 - percentage_missing) / 100))
        return f'#{red:02x}{green:02x}00'

    def get_color_tag(self, percentage_missing):
        """Determine which color tag to use based on missing percentage."""
        for (lower, upper), tag in self.color_tags.items():
            if lower <= percentage_missing < upper:
                return tag
        return self.color_tags.get((90, 100))  # Default to the highest bin for 100%

    def apply_colors_to_calendar(self):
        """Apply colors to the calendar based on missing data percentage per day."""
        # Remove existing events to prevent duplication
        self.cal.calevent_remove('all')

        # Group data by date
        grouped = self.df.groupby('date')

        for date, day_data in grouped:
            percentage_missing = self.calculate_percentage_missing(day_data)
            day_color_tag = self.get_color_tag(percentage_missing)

            # Create a calendar event with the appropriate color tag
            self.cal.calevent_create(date, f"Missing: {percentage_missing:.1f}%", day_color_tag)

    def show_day_info(self):
        """Show details of the selected day."""
        selected_date = self.cal.get_date()
        selected_date = pd.to_datetime(selected_date).date()
        day_data = self.df[self.df['date'] == selected_date]

        if not day_data.empty:
            percentage_missing = self.calculate_percentage_missing(day_data)
            biggest_gap_start, biggest_gap_end = self.calculate_biggest_gap(day_data)
            start_time = day_data['start'].min().strftime('%H:%M')
            end_time = day_data['end'].max().strftime('%H:%M')

            # Format the gap times to 12-hour format with AM/PM
            biggest_gap_start_str = biggest_gap_start.strftime('%I:%M %p')
            biggest_gap_end_str = biggest_gap_end.strftime('%I:%M %p')

            # Show info in a message box
            messagebox.showinfo(
                f"Details for {selected_date}",
                f"Missing: {percentage_missing:.1f}%\n"
                f"Biggest Gap: {biggest_gap_start_str} - {biggest_gap_end_str}\n"
                f"Start Time: {start_time}\n"
                f"End Time: {end_time}"
            )
        else:
            # If no data for the selected day, consider the entire day as missing
            messagebox.showinfo(
                "Details",
                "Missing: 100.0%\n"
                "Biggest Gap: 12:00 AM - 11:59 PM\n"
                "Start Time: N/A\n"
                "End Time: N/A"
            )

    def calculate_percentage_missing(self, day_data):
        """Calculate the percentage of time missing for a specific day."""
        # Total time in a day is 24 hours = 1440 minutes
        total_minutes = 1440

        # Sort intervals by start time
        sorted_day = day_data.sort_values(by='start')

        # Calculate recorded time by summing all intervals
        recorded_time = (sorted_day['end'] - sorted_day['start']).dt.total_seconds().sum() / 60

        missing_time = total_minutes - recorded_time
        return (missing_time / total_minutes) * 100

    def calculate_biggest_gap(self, day_data):
        """Calculate the biggest gap between two recordings and return the time range."""
        # Sort intervals by start time
        sorted_day = day_data.sort_values(by='start').reset_index(drop=True)

        # Initialize variables
        biggest_gap = timedelta(minutes=0)
        biggest_gap_start = None
        biggest_gap_end = None

        # Define the start and end of the day
        day_start = datetime.combine(sorted_day['start'].dt.date.min(), datetime.min.time())
        day_end = day_start + timedelta(days=1) - timedelta(seconds=1)

        # If there are no intervals, the entire day is missing
        if sorted_day.empty:
            return (day_start, day_end)

        # Check the gap between midnight and the first interval
        first_start = sorted_day.loc[0, 'start']
        gap = first_start - day_start
        if gap > biggest_gap:
            biggest_gap = gap
            biggest_gap_start = day_start
            biggest_gap_end = first_start

        # Iterate through consecutive intervals to find gaps
        for i in range(1, len(sorted_day)):
            previous_end = sorted_day.loc[i-1, 'end']
            current_start = sorted_day.loc[i, 'start']
            gap = current_start - previous_end
            if gap > biggest_gap:
                biggest_gap = gap
                biggest_gap_start = previous_end
                biggest_gap_end = current_start

        # Check the gap between the last interval and midnight
        last_end = sorted_day.loc[len(sorted_day)-1, 'end']
        gap = day_end - last_end
        if gap > biggest_gap:
            biggest_gap = gap
            biggest_gap_start = last_end
            biggest_gap_end = day_end

        # Handle cases where there are no gaps
        if biggest_gap_start is None or biggest_gap_end is None:
            biggest_gap_start = sorted_day['start'].min()
            biggest_gap_end = sorted_day['end'].max()

        return (biggest_gap_start, biggest_gap_end)

def get_data(intID):
    # Current issue connecting to the database
    params = { 'intersection_id':  intID }
    df_type = 'calendar'
    my = MySQLConnector()

    with yaspin(Spinners.pong, text=f"Fetching data from MySQL for intersection {intID}") as sp:
        df = my.query_videoproperties(params, df_type)

    print(df)
    return df

def preprocess_data(df):
    """Preprocess the data to calculate missing time and other statistics."""
    # Convert 'start' and 'end' to datetime
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    # Extract date from 'start' (assuming 'start' and 'end' are on the same day)
    df['date'] = df['start'].dt.date

    # Optional: Ensure that 'end' is after 'start'
    df = df[df['end'] > df['start']]

    return df

if __name__ == "__main__":
    intersection_ids = [3287, 3248, 3032, 3265, 3334]

    df_list = []

    # Fetch data for each intersection and store in a list
    for id in intersection_ids:
        data = get_data(id)  # Assuming get_data returns a DataFrame
        if not data.empty:
            df_list.append(data)

    if df_list:
        # Concatenate all DataFrames into one
        df = pd.concat(df_list, ignore_index=True)
        
        # Preprocess the data
        df = preprocess_data(df)

        # Initialize the GUI
        root = tk.Tk()
        app = CalGUI(root, df)
        root.mainloop()
    else:
        print("No data fetched from the database.")
