import tkinter as tk
from tkinter import scrolledtext


class StatusWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Status Window")

        # Create a ScrolledText widget
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=20)
        self.text_area.pack(pady=10, padx=10)

        # Disable the text area for editing
        self.text_area.config(state=tk.DISABLED)

        # Create a button to add output lines for demonstration purposes
        self.add_output_button = tk.Button(self.root, text="Add Output Line", command=self.add_output_line)
        self.add_output_button.pack(pady=5)

    def add_output_line(self):
        # Enable the text area for inserting text
        self.text_area.config(state=tk.NORMAL)

        # Insert a new line of text
        self.text_area.insert(tk.END, "This is a new status line.\n")

        # Scroll to the end of the text area
        self.text_area.yview(tk.END)

        # Disable the text area again to prevent editing
        self.text_area.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    status_window = StatusWindow(root)
    root.mainloop()
