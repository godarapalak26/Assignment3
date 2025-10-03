"""
gui.py


This GUI imports TextToImage and TextClassifier from models.py .
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import time

try:
    from models import TextToImage, TextClassifier
except Exception as e:
    TextToImage = None
    TextClassifier = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


class ModelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HIT137 — AI Model Demo ")
        self.geometry("1000x600")
        self.resizable(True, True)

        self.tti = None
        self.clf = None
        self._displayed_image = None
        self._last_image_path = None

        self._build_ui()

    def _build_ui(self):
        ctrl_frame = ttk.Frame(self, padding=(10, 10))
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl_frame, text="Select Model:").pack(anchor="w")
        model_options = [
            ("Text → Image (Stable Diffusion v2.1)", "text2image"),
            ("Text Classification (DistilBERT - Sentiment)", "textclass")
        ]
        self.model_combo = ttk.Combobox(ctrl_frame, values=[t for t, _ in model_options], state="readonly")
        self.model_combo.current(0)
        self.model_combo.pack(fill="x", pady=(0, 8))
        self._model_map = {t: v for t, v in model_options}

        ttk.Label(ctrl_frame, text="Input (text prompt or text to classify):").pack(anchor="w")
        self.input_text = tk.Text(ctrl_frame, height=8, width=40, wrap="word")
        self.input_text.pack(pady=(0, 8))

        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", pady=(4, 4))
        self.run_btn = ttk.Button(btn_frame, text="Run Model", command=self._on_run)
        self.run_btn.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_frame, text="Clear", command=self._on_clear).pack(fill="x", pady=(0, 4))
        ttk.Button(btn_frame, text="Save Last Output As...", command=self._on_save_output).pack(fill="x", pady=(0, 4))
        ttk.Button(btn_frame, text="Exit", command=self.destroy).pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl_frame, textvariable=self.status_var, foreground="blue").pack(anchor="w", pady=(8, 0))

        out_frame = ttk.Frame(self, padding=(10, 10))
        out_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(out_frame, text="Image Output:").pack(anchor="w")
        self.canvas = tk.Canvas(out_frame, bg="grey", height=320)
        self.canvas.pack(fill="both", expand=True)

        ttk.Label(out_frame, text="Model Output (text):").pack(anchor="w", pady=(6, 0))
        self.output_text = tk.Text(out_frame, height=8, wrap="word")
        self.output_text.pack(fill="x", pady=(0, 8))

        ttk.Label(out_frame, text="Model Info / Notes:").pack(anchor="w")
        self.info_label = tk.Label(out_frame, text="Select a model and enter input text.")
        self.info_label.pack(anchor="w")

        self.after(100, self._check_import)

    def _check_import(self):
        if IMPORT_ERROR is not None:
            messagebox.showerror("Import Error", f"Failed to import models.py: {IMPORT_ERROR}")
            self.status_var.set("Import error — see popup")

    def _on_run(self):
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Running model...")
        t = threading.Thread(target=self._run_model_thread, daemon=True)
        t.start()

    def _run_model_thread(self):
        try:
            sel_display = self.model_combo.get()
            model_id = self._model_map.get(sel_display)
            text_input = self.input_text.get("1.0", tk.END).strip()
            if not text_input:
                raise ValueError("Input box is empty.")

            if model_id == "text2image":
                if self.tti is None:
                    self.status_var.set("Loading TextToImage model...")
                    self.tti = TextToImage()
                out_dir = "outputs"
                os.makedirs(out_dir, exist_ok=True)
                timestamp = int(time.time())
                save_path = os.path.join(out_dir, f"tti_output_{timestamp}.png")
                path = self.tti.generate_image(text_input, save_path=save_path)
                self._last_image_path = path
                self._display_image_on_canvas(path)
                self._append_output_text(f"Image saved to: {path}")

            elif model_id == "textclass":
                if self.clf is None:
                    self.status_var.set("Loading TextClassifier model...")
                    self.clf = TextClassifier()
                result = self.clf.classify(text_input)
                out_str = f"Label: {result.get('label')}\nConfidence: {result.get('confidence')}"
                self._append_output_text(out_str)

            self.status_var.set("Done.")
        except Exception as e:
            self._append_output_text(f"Error: {e}")
            messagebox.showerror("Model Error", str(e))
            self.status_var.set("Error occurred.")
        finally:
            self.run_btn.config(state=tk.NORMAL)

    def _append_output_text(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def _display_image_on_canvas(self, path):
        try:
            img = Image.open(path)
        except Exception as e:
            self._append_output_text(f"Failed to open generated image: {e}")
            return
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width() or 600
        ch = self.canvas.winfo_height() or 320
        img.thumbnail((cw, ch))
        self._displayed_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self._displayed_image)
        self._append_output_text("Image displayed in GUI.")

    def _on_clear(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.canvas.delete("all")
        self._displayed_image = None
        self._last_image_path = None
        self.status_var.set("Cleared.")

    def _on_save_output(self):
        if self._last_image_path and os.path.exists(self._last_image_path):
            save_to = filedialog.asksaveasfilename(defaultextension=".png")
            if save_to:
                with open(self._last_image_path, "rb") as fr, open(save_to, "wb") as fw:
                    fw.write(fr.read())
                messagebox.showinfo("Saved", f"Image saved to {save_to}")
        else:
            text = self.output_text.get("1.0", tk.END).strip()
            if not text:
                messagebox.showinfo("Nothing to save", "No output to save.")
                return
            save_to = filedialog.asksaveasfilename(defaultextension=".txt")
            if save_to:
                with open(save_to, "w", encoding="utf-8") as fw:
                    fw.write(text)
                messagebox.showinfo("Saved", f"Text output saved to {save_to}")


if __name__ == "__main__":
    app = ModelApp()
    app.mainloop()

"""
Model 1 – Stable Diffusion v2.1
Publisher: Stability AI
Category: Text-to-Image
Description: Generates images from text descriptions.
Why chosen: Demonstrates multimodal AI capability.

Model 2 – DistilBERT Sentiment Classifier
Publisher: Hugging Face
Category: Text Classification
Description: Analyzes text sentiment (Positive/Negative).
Why chosen: Lightweight, accurate, and demonstrates NLP integration.
"""

