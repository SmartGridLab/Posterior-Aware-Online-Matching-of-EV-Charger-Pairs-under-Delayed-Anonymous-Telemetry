import pandas as pd
import numpy as np


def get_command_data(initial_timestamp, command_values, tau=60):
    """Build the command-value dataframe."""
    timestamps = pd.date_range(start=initial_timestamp, periods=len(command_values), freq=f'{tau}s')
    df_command = pd.DataFrame({
        "timestamp": timestamps,
        "command_value": command_values
    })
    return df_command

def _infer_tau_seconds(df_command) -> int:
    if len(df_command) >= 2:
        dt = (df_command.at[1, "timestamp"] - df_command.at[0, "timestamp"]).total_seconds()
        if np.isfinite(dt) and dt > 0:
            return int(round(dt))
    return 60


def generate_charger_current(df_command):
    """Generate the charger current trace from the command values."""
    results = []
    charger_current = 0.0  # initial value
    current_timestamp = df_command.at[0, 'timestamp']
    tau_s = _infer_tau_seconds(df_command)

    for i in range(len(df_command)):
        if i == 0:
            next_timestamp = df_command.at[i + 1, 'timestamp']
            while current_timestamp < next_timestamp:
                current_timestamp += pd.Timedelta(seconds=1)
                results.append((current_timestamp, charger_current))
        else:
            # (1) start of the change
            if i == 1:
                change_start_time = df_command.at[i, 'timestamp'] + pd.Timedelta(seconds=15)  # delay before the change starts
            else:
                change_start_time = df_command.at[i, 'timestamp'] + pd.Timedelta(seconds=5)

            while current_timestamp < change_start_time:
                current_timestamp += pd.Timedelta(seconds=1)
                results.append((current_timestamp, charger_current))

            # (2) change in progress
            current_offset = 0.0221 * df_command.at[i, 'command_value'] + 0.2831
            target_current = df_command.at[i, 'command_value'] - current_offset

            # command value increases
            if charger_current < target_current:
                command_diff = df_command.at[i, 'command_value'] - df_command.at[i - 1, 'command_value']
                delta_current_per_sec = 0.0327 * command_diff + 0.3787

                while charger_current < target_current:
                    charger_current = min(charger_current + delta_current_per_sec, target_current)
                    current_timestamp += pd.Timedelta(seconds=1)
                    results.append((current_timestamp, charger_current))
            # command value decreases
            else:
                while charger_current > target_current:
                    charger_current = max(charger_current - 4, target_current)
                    current_timestamp += pd.Timedelta(seconds=1)
                    results.append((current_timestamp, charger_current))

            # (3) end of the change
            if i < len(df_command) - 1:
                next_timestamp = df_command.at[i + 1, 'timestamp']
                while current_timestamp < next_timestamp:
                    current_timestamp += pd.Timedelta(seconds=1)
                    results.append((current_timestamp, charger_current))
            else:
                end_time = df_command.at[i, 'timestamp'] + pd.Timedelta(seconds=int(tau_s))
                while current_timestamp < end_time:
                    current_timestamp += pd.Timedelta(seconds=1)
                    results.append((current_timestamp, charger_current))

    # convert the results to a dataframe
    df_charger_generated = pd.DataFrame(results, columns=['timestamp', 'charger_current'])
    df_charger_generated.set_index('timestamp', inplace=True)
    df_charger_generated['charger_current'] = df_charger_generated['charger_current'].round(1)
    return df_charger_generated
