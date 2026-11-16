import numpy as np
import random

wbk_color_dict = {"green": np.array([0, 150, 130]),
                  "red": np.array([178, 55, 44]),
                  "yellow": np.array([238, 183, 13]),
                  "blue": np.array([20, 68, 102]),
                  "grey": np.array([59, 79, 88]),
                  "turquois": np.array([0, 131, 143]),
                  "l_gray": np.array([196, 208, 207]),
                  "l_blue": np.array([60, 152, 217]),
                  "l_green": np.array([39, 255, 226])}


def get_random_color(exclude_list):
    # Filter the dictionary to exclude specified colors
    available_colors = {k: v for k, v in wbk_color_dict.items() if k not in exclude_list}

    if not available_colors:
        # Generate a random color if all are excluded
        random_color = np.random.randint(0, 256, size=3)
        random_color_name = f"random_color_{random_color[0]}_{random_color[1]}_{random_color[2]}"
        return random_color_name, random_color

    # Select a random color from the available colors
    color_name = random.choice(list(available_colors.keys()))
    return color_name, available_colors[color_name]