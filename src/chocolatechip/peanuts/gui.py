import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
import pandas as pd
from yaspin import yaspin
from yaspin.spinners import Spinners
from chocolatechip.MySQLConnector import MySQLConnector

class CalGUI:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        self.root.title("Calendar")
        self.root.geometry("400x400")
        self.root.resizable(False, False)

        # Calendar widget
        self.cal = Calendar(self.root, selectmode="day")
        self.cal.pack(pady=20)

        # Button to fetch the details for the selected day
        self.button = ttk.Button(self.root, text="See Insights", command=self.show_day_info)
        self.button.pack(pady=20)

        # Apply colors to the calendar based on missing percentage
        self.apply_colors_to_calendar()

    def apply_colors_to_calendar(self):
        """Apply colors to the calendar based on missing data percentage per day."""
        for date, day_data in self.df.groupby(self.df['timestamp'].dt.date):
            percentage_missing = self.calculate_percentage_missing(day_data)
            day_color = self.calculate_day_color(percentage_missing)
            
            # Convert date to string format recognized by tkcalendar
            date_str = date.strftime('%Y-%m-%d')
            
            # Create a calendar event with the color representing missing data
            self.cal.calevent_create(date, f"Missing: {percentage_missing}%", 'custom')
            self.cal.tag_config('custom', background=day_color)

    def calculate_day_color(self, percentage_missing):
        """Convert percentage missing to a color from red to green."""
        red = int(255 * (percentage_missing / 100))
        green = int(255 * ((100 - percentage_missing) / 100))
        return f'#{red:02x}{green:02x}00'

    def show_day_info(self):
        """Show details of the selected day."""
        selected_date = self.cal.get_date()
        day_data = self.df[self.df['timestamp'].dt.date == pd.to_datetime(selected_date).date()]

        if not day_data.empty:
            percentage_missing = self.calculate_percentage_missing(day_data)
            biggest_gap = self.calculate_biggest_gap(day_data)
            start_time = day_data['timestamp'].min().strftime('%H:%M')
            end_time = day_data['timestamp'].max().strftime('%H:%M')

            # Show info in a message box
            # May want to change to show biggest time gap hours instead of size of biggest gap
            messagebox.showinfo(f"Details for {selected_date}",
                                f"Missing: {percentage_missing}%\n"
                                f"Biggest Gap: {biggest_gap} minutes\n"
                                f"Start Time: {start_time}\n"
                                f"End Time: {end_time}")
        else:
            messagebox.showinfo("Details", "No data available for this day.")

    def calculate_percentage_missing(self, day_data):
        """Calculate the percentage of time missing for a specific day."""
        # Total time in a day is 24 hours = 1440 minutes
        total_minutes = 1440

        # Calculate the actual recorded time (sum of the gaps)
        recorded_time = day_data['timestamp'].diff().sum().total_seconds() / 60

        missing_time = total_minutes - recorded_time
        return (missing_time / total_minutes) * 100

    def calculate_biggest_gap(self, day_data):
        """Calculate the biggest gap between two recordings."""
        gaps = day_data['timestamp'].diff() / pd.Timedelta(minutes=1)
        return gaps.max() if not gaps.empty else 0

def get_data(intID):
    # Current issue connecting to the database
    params = { 'intersection_id':  intID }
    df_type = 'calendar'
    my = MySQLConnector()

    with yaspin(Spinners.pong, text=f"Fetching data from MySQL for intersection {intID}") as sp:
        df = my.handleRequest(params, df_type)

    print(df)
    return df

def preprocess_data(df):
    """Preprocess the data to calculate missing time and other statistics."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Here, you could also calculate statistics per day and attach them to the dataframe
    return df

if __name__ == "__main__":
    # Sample DataFrame for testing
    intersection_ids = [3287, 3248, 3032, 3265, 3334]

    df_list = []

    # Fetch data for each intersection and store in a list
    for id in intersection_ids:
        data = get_data(id)  # Assuming get_data returns a DataFrame
        df_list.append(data)

    # Concatenate all DataFrames into one
    df = pd.concat(df_list, ignore_index=True)
    
    # Preprocess the data
    df = preprocess_data(df)

    root = tk.Tk()
    app = CalGUI(root, df)
    root.mainloop()
