import tkinter as tk
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from gui import SecurityCamApp

def main():
    root = tk.Tk()
    root.geometry("1200x700")
    root.minsize(900, 500)
    app = SecurityCamApp(root)
    root.after(500, app.start)
    root.mainloop()

if __name__ == "__main__":
    main()
