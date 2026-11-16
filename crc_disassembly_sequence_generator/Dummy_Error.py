import tkinter as tk
from tkinter import messagebox
import time

# Erstelle das Hauptfenster
root = tk.Tk()
root.withdraw()  # Verstecke das Hauptfenster

time.sleep(5)

root.attributes("-topmost", True)

# Zeige eine Warnmeldung
messagebox.showerror("Error", "No Connection to Gripper!")



# Starte die Event-Schleife
root.mainloop()
