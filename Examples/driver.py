import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function

def run():
    water = np.loadtxt("water_density_vs_temp_usgs.txt")
    air = np.loadtxt("air_density_vs_temp_usgs.txt")

    xd_w, yd_w = water[:,0], water[:,1]
    xd_a, yd_a = air[:,0], air[:,1]

    temps_w = np.linspace(xd_w[0], xd_w[-1], 100)
    temps_a = np.linspace(xd_a[0], xd_a[-1], 100)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for row, (xd, yd, temps, title) in enumerate([
        (xd_w, yd_w, temps_w, "Water Density"),
        (xd_a, yd_a, temps_a, "Air Density")
    ]):
        for order in (1, 2, 3):
            f = spline_function(xd, yd, order=order)
            y_interp = f(temps)

            ax = axes[order-1, row]
            ax.plot(xd, yd, 'o', label="data")
            ax.plot(temps, y_interp, '-', label=f"spline order {order}")
            ax.set_title(f"{title} - order {order}")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig("spline_plots.png")
    plt.show()

if __name__ == "__main__":
    run()