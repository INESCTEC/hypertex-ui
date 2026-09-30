#!/usr/bin/env python3
import tkinter as tk
from tkinter import *
from tkinter import messagebox, ttk, filedialog
import tkinter.font as tkFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import cv2
from PIL import Image, ImageTk
import os
import threading
from utils import *

import spectral

IMG_LEN = 800
CURRENT_DIR = os.path.dirname(__file__)

class SpecGUI:

    def __init__(self):
        self._resize_after_id = None   

        self.window = tk.Tk()
        ico         = Image.open(os.path.join(CURRENT_DIR, 'images', 'logo.png'))
        photo       = ImageTk.PhotoImage(ico)
        self.window.wm_iconphoto(False, photo)
        self.window.title('HyperTex GUI')

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}+0+0")

        self.window.bind("<Configure>", self.on_window_resize)
        self.window.protocol("WM_DELETE_WINDOW", self.buttonExit)

        self.window.grid_columnconfigure(0, weight=10)
        self.window.grid_rowconfigure(0, weight=10)

        inesctec_logo = Image.open(os.path.join(CURRENT_DIR, 'images', 'inesctec-logo.png')).resize((285, 40),Image.LANCZOS)
        self.inesctec_tk = ImageTk.PhotoImage(image=inesctec_logo)
        self.inesctec_lbl = Label(self.window, image=self.inesctec_tk, borderwidth=0)
        self.inesctec_lbl.place(x=30, y=5)

        # HDF5 & Ground Truth (GT) state variables
        self.is_h5 = False
        self.h5_path = None
        self.current_gt_mask = None
        self.cb_show_gt = IntVar(value=0)
        
        # Button Exit
        self.buttonExitProgram = tk.Button(
        text=" Exit",
        justify=CENTER,
        width=4,
        height=1,
        bg="indian red",
        fg="black",
        command=self.buttonExit,
        )
        self.buttonExitProgram.grid(row=0, column=9, columnspan=2, sticky='ne', padx=(200, 0), pady=0)

        self.buttonJson2Npy = tk.Button(
            self.window,
            text="JSON → NPY",
            width=10,
            bg="lightgray",
            command=self.buttonJsonToNpy
        )

        self.buttonDataset2HDF5 = tk.Button(
            self.window,
            text="Dataset → HDF5",
            width=12,
            bg="lightgray",
            command=self.buttonDatasetToHDF5
        )

        # Button Set Directory
        self.buttonDir = tk.Button(
        text="Set Directory",
        width=10,
        height=1,
        bg="lightgray",
        command=self.buttonDirectory,
        )
        self.buttonDir.grid(row=8, column=9, sticky='e', padx=140)

        # Button Set HDF5
        self.buttonH5 = tk.Button(
        text="Set HDF5 File",
        width=10,
        height=1,
        bg="lightgray",
        # fg="darkgray",
        command=self.buttonHDF5,
        )
        self.buttonH5.grid(row=8, column=9, sticky='e', padx=240)



        self.reflectance_plot = HSIPlot(self.window, title="Reflectance", ylabel="Reflectance", ylim=(-0.1, 1.2))
        self.raw_plot = HSIPlot(self.window, title="Raw sensor signal", ylabel="Raw signal", ylim=(-100, 4100))

        self.classification_table = ClassificationTable(self.window)

        bands_fx17 = [float(band) for band in open(os.path.join(CURRENT_DIR,"bands_fx17.txt"), "r").read().splitlines()]
        self.fx17_image = HSICanvasImage(self.window, width=640, height=IMG_LEN,
                                         result_table=self.classification_table,
                                         data_plots={'reflectance': self.reflectance_plot, 'raw': self.raw_plot},
                                         line_color='red', bands=bands_fx17, rgb_bands=(19,149,219), root_gui=self)
        self.fx17_image.place(x=30, y=70)

        self.window.update()

        # Treeview for Directory
        columns = ('capture_name', 'header_name', 'label_name')
        self.tree_capt = ttk.Treeview(
            self.window, columns=columns, displaycolumns=columns, show='headings', height=25,
        )
        self.tree_capt.column('#0', width=0, minwidth=0, stretch=NO)
        self.tree_capt.heading('capture_name', text='Capture Name')
        self.tree_capt.column("capture_name", minwidth=0, width=300, stretch=NO)
        self.tree_capt.heading('header_name', text='Header')
        self.tree_capt.column("header_name", minwidth=0, width=50, stretch=NO)
        self.tree_capt.heading('label_name', text='Label')
        self.tree_capt.column("label_name", minwidth=0, width=50, stretch=NO)

        self.absolute_path = os.path.dirname(__file__)

        self.captures_dir = os.path.join(self.absolute_path, "data")
        directory_data = directory_loader(self.captures_dir)
        directory_data.sort()

        for data in directory_data:
            self.tree_capt.insert('', tk.END, values=data)   

        self.tree_capt.bind('<<TreeviewSelect>>', self.item_selected)
        self.tree_capt.grid_forget()

        self.tree_capt.place(relx=1.0, rely=1.0, x=-25, y=-190, anchor='se')

        scrollbar = ttk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.tree_capt.yview)
        self.tree_capt.configure(yscroll=scrollbar.set)
        scrollbar.place(in_=self.tree_capt, relx=1.0, x=-15, y=0, relheight=1.0)

        self.tree_capt_info = Label(self.window, font=('Calibri 10'))
        
        self.buttonDir.place(in_=self.tree_capt, x=206, y=-30)
        self.buttonH5.place(in_=self.tree_capt, x=306, y=-30)
        self.tree_capt_info.place(in_=self.tree_capt, x=0, relx=0.5, rely=1.0, y=10, anchor=CENTER)
        
        self.buttonJson2Npy.place(in_=self.tree_capt, relx=0.33, y=-100, anchor=CENTER)
        self.buttonDataset2HDF5.place(in_=self.tree_capt, relx=0.66, y=-100, anchor=CENTER)
        

        self.cb_show_gt = IntVar(value=0)
        self.show_gt_check_button = ttk.Checkbutton(self.window,
                        text='Show GT',
                        variable=self.cb_show_gt,
                        onvalue=True, offvalue=False, command=self.toggle_gt_display)


        self.show_gt_check_button.place(in_=self.tree_capt, x=0, y=-20)

        self.window.update()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ↓ ↓ ↓ ↓   Functions   ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓   Functions   ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓   Functions  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓   Functions   ↓ ↓ ↓ ↓  # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    def item_selected(self, event=None):
        if not self.tree_capt.selection(): return

        self.tree_capt_info.configure( text="Loading sample...", bg="gold" )
                
        self.window.update_idletasks()
        
        for selected_item in self.tree_capt.selection():
            item = self.tree_capt.item(selected_item)
            record = item['values']
            sample_name = str(record[0])

            capture_hsi = None
            raw_hsi = None
            gt_data = None
            labelme_json_path = None

            if self.is_h5 and self.h5_path:
                capture_hsi, raw_hsi, gt_data = load_hdf5_sample_data(self.h5_path, sample_name)
            else:
                capture_dir = os.path.join(self.captures_dir, sample_name, "capture")
                reflectance_path = os.path.join(capture_dir, 'REFLECTANCE_' + sample_name + '.hdr')
                raw_path = os.path.join(capture_dir, sample_name + '.hdr')
                
                if not os.path.exists(reflectance_path):
                    candidates = [p for p in os.listdir(capture_dir)] if os.path.isdir(capture_dir) else []
                    reflectance_path = next((os.path.join(capture_dir, p) for p in candidates
                                            if p.startswith('REFLECTANCE_') and p.endswith('.hdr')), raw_path)
                if os.path.exists(reflectance_path):
                    capture_hsi = spectral.open_image(reflectance_path)
                if os.path.exists(raw_path) and os.path.abspath(raw_path) != os.path.abspath(reflectance_path):
                    raw_hsi = spectral.open_image(raw_path)
                gt_data = detect_sample_gt(self.captures_dir, sample_name)

            annotated_pixels = 0

            if gt_data is not None:
                annotated_pixels = np.count_nonzero(np.any(gt_data > 0, axis=-1))

                labelme_json_path = find_sample_labelme_json(self.captures_dir, sample_name)

            if capture_hsi is not None:
                self.capture_hsi = capture_hsi
                self.current_gt_mask = gt_data
                rgb_bands = self.fx17_image.get_selected_rgb_indices()
                capture_img = spectral.get_rgb(capture_hsi, bands=rgb_bands)
                if capture_img.dtype != np.uint8:
                    if np.max(capture_img) <= 1.0:
                        capture_img = (capture_img * 255).astype(np.uint8)
                    else:
                        capture_img = capture_img.astype(np.uint8)

                self.fx17_image.set_gt_data(self.current_gt_mask, show_gt=bool(self.cb_show_gt.get()))
                self.fx17_image.set_labelme_annotation(labelme_json_path)
                self.fx17_image.set_input_data(capture_hsi, capture_img, raw_hsi=raw_hsi)

                info_txt = f'Annotated Pixels: {annotated_pixels:,}'
                
                self.tree_capt_info.configure(text=info_txt, bg="light gray")
            else:
                self.tree_capt_info.configure(text='! Image Not Found !', bg="yellow")

    def setDefault(self):
        self.tree_capt_info.configure(text='', bg="light gray")
    
    def buttonDirectory(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:             
            self.is_h5 = False
            self.h5_path = None
            self.tree_capt.delete(*self.tree_capt.get_children())
            self.captures_dir = folder_selected
            directory_data = directory_loader(self.captures_dir)
            directory_data.sort()

            for data in directory_data:
                self.tree_capt.insert('', tk.END, values=data) 
            self.tree_capt_info.configure(text=f'Directory: {os.path.basename(folder_selected)}', bg="light gray")

    def buttonHDF5(self):
        file_selected = filedialog.askopenfilename(
            title="Select HDF5 Dataset",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")]
        )
        if file_selected:
            self.is_h5 = True
            self.h5_path = file_selected
            self.captures_dir = os.path.dirname(file_selected)
            self.tree_capt.delete(*self.tree_capt.get_children())
            directory_data = hdf5_directory_loader(self.h5_path)
            for data in directory_data:
                self.tree_capt.insert('', tk.END, values=data)
            self.tree_capt_info.configure(text=f'HDF5: {os.path.basename(file_selected)}', bg="light green")

    def toggle_gt_display(self):
        show_gt = bool(self.cb_show_gt.get())
        if hasattr(self, 'fx17_image'):
            self.fx17_image.toggle_gt(show_gt)

    def on_window_resize(self, event):
        if event.widget != self.window:
            return
        
        win_w = event.width
        win_h = event.height
        if win_w < 400 or win_h < 400:
            return

        canvas_w = min(640, int(win_w * 0.35))
        canvas_h = min(IMG_LEN, int(win_h * 0.75))

        margin = 30

        if hasattr(self, 'fx17_image') and self.fx17_image is not None:
            self.fx17_image.resize_canvas(canvas_w, canvas_h)
            self.fx17_image.place(x=margin, y=70)

        if hasattr(self, 'classification_table') and self.classification_table is not None:
            table_h = 40
            self.classification_table.place( x=margin + canvas_w, y=win_h - table_h - 30, anchor=CENTER)

        if hasattr(self, 'reflectance_plot') and self.reflectance_plot is not None:
            plot_x = canvas_w + 80
            plot_w = canvas_w
            plot_h = (canvas_h - 20) //2
            self.reflectance_plot.place( x=plot_x, y=70, width=plot_w, height=plot_h)
            self.raw_plot.place(x=plot_x, y=70 + plot_h + 20, width=plot_w, height=plot_h)

    def buttonExit(self):
        plt.close('all')
        self.window.quit()
        self.window.destroy()
   
    def buttonJsonToNpy(self):
        if not messagebox.askyesno(
            "Confirm",
            "Convert all LabelMe JSON annotations into NPY files from the current directory?"
            ):
            return
        
        def worker():
            try:
                extended_data_loader_paths(
                    self.captures_dir,
                    re_save=True
                )

                self.window.after(
                    0,
                    lambda: self.tree_capt_info.configure(
                        text="NPY generation complete",
                        bg="light green"
                    )
                )

            except Exception as e:
                self.window.after(
                    0,
                    lambda: self.tree_capt_info.configure(
                        text="NPY generation failed",
                        bg="indian red"
                    )
                )
                print(e)
            self.window.after(2000, self.setDefault)

        self.tree_capt_info.configure(
            text="Generating NPY files...",
            bg="gold"
        )

        threading.Thread(target=worker, daemon=True).start()

    def buttonDatasetToHDF5(self):

        if not messagebox.askyesno(
            "Confirm",
            "Create an HDF5 dataset from the current directory?"
            ):
            return
        
        selected_samples = self.selectSamplesForHDF5()

        if not selected_samples:
            return

        output_file = filedialog.asksaveasfilename(
            title="Save HDF5 Dataset",
            defaultextension=".hdf5",
            filetypes=[("HDF5 Dataset", "*.hdf5")]
        )

        if not output_file:
            return

        def worker():
            try:
                dataset_to_hdf5(
                    self.captures_dir,
                    os.path.splitext(output_file)[0]
                )

                self.window.after(
                    0,
                    lambda: self.tree_capt_info.configure(
                        text="HDF5 dataset created",
                        bg="light green"
                    )
                )

            except Exception as e:
                self.window.after(
                    0,
                    lambda: self.tree_capt_info.configure(
                        text="HDF5 creation failed",
                        bg="indian red"
                    )
                )
                print(e)
            self.window.after(2000, self.setDefault)

        self.tree_capt_info.configure(
            text="Creating HDF5 dataset...",
            bg="gold"
        )

        threading.Thread(target=worker, daemon=True).start()

    def selectSamplesForHDF5(self):
        result = []
        popup = tk.Toplevel(self.window)
        popup.title("Select Samples for HDF5")
        popup.geometry("500x600")
        popup.transient(self.window)
        popup.grab_set()

        tk.Label(
            popup,
            text="Select the samples to include in the HDF5 dataset",
        ).pack(pady=5)

        frame = tk.Frame(popup)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(
            frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
            justify="center",
        )

        scrollbar.config(command=listbox.yview)

        listbox.pack(fill="both", expand=True)

        samples = []

        for item in self.tree_capt.get_children():
            sample = self.tree_capt.item(item)["values"][0]
            samples.append(sample)
            listbox.insert(tk.END, sample)

        def select_all():
            listbox.select_set(0, tk.END)

        def clear_all():
            listbox.selection_clear(0, tk.END)

        def confirm():

            selected = [
                listbox.get(i)
                for i in listbox.curselection()
            ]

            if not selected:
                messagebox.showwarning(
                    "No Selection",
                    "Please select at least one sample."
                )
                return

            popup.destroy()

            output_file = filedialog.asksaveasfilename(
                title="Save HDF5 Dataset",
                defaultextension=".hdf5",
                filetypes=[("HDF5 Dataset", "*.hdf5")]
            )

            if not output_file:
                return

            self.createHDF5FromSelection(
                output_file,
                selected
            )

        btn_frame = tk.Frame(popup)
        btn_frame.pack(fill="x", pady=5)

        tk.Button(
            btn_frame,
            text="Select All",
            command=select_all
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Clear",
            command=clear_all
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Create HDF5",
            command=confirm
        ).pack(side="right", padx=5)

        self.window.wait_window(popup)
        return result


    def createHDF5FromSelection(self, output_file, selected_samples):
        def worker():

            dataset_txt = os.path.splitext(output_file)[0] + "_dataset.txt"

            try:

                with open(dataset_txt, "w") as f:
                    f.write("\n".join(selected_samples))

                dataset_to_hdf5( self.captures_dir, os.path.splitext(output_file)[0], dataset_txt )

                self.window.after(
                    0,
                    lambda: self.tree_capt_info.configure(
                        text="HDF5 dataset created",
                        bg="light green"
                    )
                )

                self.window.after(2000, self.setDefault)

            except Exception as e:

                self.window.after(
                    0,
                    lambda: self.tree_capt_info.configure(
                        text="HDF5 creation failed",
                        bg="indian red"
                    )
                )

                print(e)

            finally:

                if os.path.exists(dataset_txt):
                    os.remove(dataset_txt)

        self.tree_capt_info.configure(text="Creating HDF5 dataset...", bg="gold")

        threading.Thread(target=worker, daemon=True).start()
        
    def start(self):
        self.window.mainloop()
    

class HSICanvasImage(tk.Canvas):
    def __init__(self, root, width=800, height=640, line_color='red', result_table=None, data_plots=None, bands=None, rgb_bands=None, root_gui=None):
        super().__init__(root, width=width, height=height, borderwidth=0, highlightthickness=0)
        self.root = root
        self.root_gui = root_gui
        
        self.image = Image.fromarray(np.zeros((height,width), dtype="uint8"))
        self.tk_image = ImageTk.PhotoImage(image=self.image)

        # Draw image on canvas
        self.canvas_image = self.create_image(0, 0, anchor='nw', image=self.tk_image)

        # Initial line positions (centered)
        self.v_line_x = 1
        self.h_line_y = 1

        # Draw lines
        self.v_line = self.create_line(self.v_line_x, 0, self.v_line_x, self.image.height, fill=line_color, width=1)
        self.h_line = self.create_line(0, self.h_line_y, self.image.width, self.h_line_y, fill=line_color, width=1)

        # Bind mouse events
        self.bind("<ButtonPress-1>", self.on_left_click_down)
        self.bind("<B1-Motion>", self.on_motion, add="+")
        self.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind('<Motion>',  self.on_motion)
        self.bind('<Leave>',  self.on_leave)
        self.bind('<Button-3>',  self.on_right_click_down)
        self.bind('<Double-Button-1>',  self.on_double_click)

        self.status = tk.Label(root, text="")

        self.capture_hsi    = np.empty(1)
        self.raw_hsi        = None
        self.capture_image  = np.empty(1)
        self.result_hsi     = np.empty(1)
        self.result_image   = np.empty(1)
        self.overlap_result = np.empty(1)

        self.gt_mask        = None
        self.gt_rgb         = None
        self.labelme_json_path = None
        self.show_gt        = False

        self.result_table   = result_table
        self.data_plots     = data_plots or {}
        self.rgb_bands = rgb_bands

        self.plot_names = {}
        if bands is not None:
            for plot_type, plot in self.data_plots.items():
                self.plot_names[plot_type] = plot.add_bands_configuration(bands=bands, color=line_color)
                plot.add_draggable_lines(min(bands), max(bands), name=self.plot_names[plot_type], rgb_bands=self.rgb_bands)
    
    def set_gt_data(self, gt_mask, show_gt=False):
        self.gt_mask = gt_mask
        self.gt_rgb = gt_to_colored_rgb(gt_mask) if gt_mask is not None else None
        self.show_gt = show_gt

    def set_labelme_annotation(self, json_path):
        self.labelme_json_path = json_path

    def toggle_gt(self, show_gt):
        self.show_gt = show_gt
        if self.capture_image.size > 1:
            self.update_image_view(img_rgb=self.capture_image)

    def resize_canvas(self, new_width, new_height):
        if new_width <= 0 or new_height <= 0:
            return
        self.config(width=new_width, height=new_height)
        self.image = Image.fromarray(np.zeros((new_height, new_width), dtype="uint8"))
        if self.capture_image.size > 1:
            self.update_image_view(img_rgb=self.capture_image)


    def _get_image_coords_from_canvas(self, canvas_x, canvas_y):
        img_h, img_w = self.capture_hsi.shape[:2]
        canvas_w = self.winfo_width() if self.winfo_width() > 10 else int(self["width"])
        canvas_h = self.winfo_height() if self.winfo_height() > 10 else int(self["height"])

        scale_x = img_w / max(1, canvas_w)
        scale_y = img_h / max(1, canvas_h)

        img_x = int(canvas_x * scale_x)
        img_y = int(canvas_y * scale_y)

        return img_x, img_y
    
    def place(self, x, y):
        super().place(x=x, y=y)
        self.root.update()
        self.status.place(x=(self.winfo_width() + self.winfo_x()), y=(self.winfo_height())  + self.winfo_y() + 3, anchor='ne')

    def on_left_click_down(self, event):
        self.v_line_x = 1 if event.x < 0 else event.x
        self.h_line_y = 1 if event.y < 0 else event.y
        
    def on_mouse_up(self, event):
        self.dragging = None

    def on_motion(self, event):
        _coords_x, _coords_y = event.x, event.y
        gt_info_str = ""

        try:
            _coords_x, _coords_y = self._get_image_coords_from_canvas(event.x, event.y)
            if not (0 <= _coords_y < self.capture_hsi.shape[0] and 0 <= _coords_x < self.capture_hsi.shape[1]):
                return

            self.move_lines(event.x, event.y)

            if self.gt_mask is not None:
                _, _, gt_vec = get_pixel_gt_info(self.gt_mask, _coords_y, _coords_x)
                if self.result_table is not None and gt_vec is not None:
                    self.result_table.insert_data(gt_vec, as_percentage=True)

            original_labels = labelme_pixel_annotation(
                self.labelme_json_path, _coords_x, _coords_y)
            if original_labels:
                original_text = ', '.join(f'{label}: {percentage:g}%' for label, percentage in original_labels)
                gt_info_str = f'JSON GT = {original_text} | '
        
        except Exception as e:
            pass
        self.status["text"] = f'{gt_info_str}({_coords_y},{_coords_x})'

    def on_leave(self, event):
        self.status["text"] = ''
        self.move_lines(self.v_line_x, self.h_line_y)
        if self.result_table is not None: self.result_table.reset_data()

    def on_right_click_down(self, event):
        self.v_line_x = 1
        self.h_line_y = 1

        for plot_type, plot in self.data_plots.items():
            plot.clear_plot(self.plot_names[plot_type])
        self.delete('oval')

    def on_double_click(self, event):
        if self.data_plots:
            try:
                img_x, img_y = self._get_image_coords_from_canvas(event.x, event.y)
                if not (0 <= img_y < self.capture_hsi.shape[0] and 0 <= img_x < self.capture_hsi.shape[1]):
                    return
                color = self.data_plots['reflectance'].set_plot_data(
                    self.capture_hsi[img_y, img_x], name=self.plot_names['reflectance'])
                if self.raw_hsi is not None:
                    self.data_plots['raw'].set_plot_data(
                        self.raw_hsi[img_y, img_x], name=self.plot_names['raw'], color=color)

                x1, y1 = (event.x - 3), (event.y - 3)
                x2, y2 = (event.x + 3), (event.y + 3)
                self.create_oval(x1, y1, x2, y2, width=1, fill=color, tags='oval')
            except Exception as e:
                print(f'Error on double_click: {e}')
                pass

    def move_lines(self, coords_x, coords_y):
        try:
            canvas_w = self.winfo_width() if self.winfo_width() > 10 else int(self["width"])
            canvas_h = self.winfo_height() if self.winfo_height() > 10 else int(self["height"])

            v_line_x = 1 if coords_x < 0 else coords_x
            self.coords(self.v_line, v_line_x, 0, v_line_x, canvas_h)
            h_line_y = 1 if coords_y < 0 else coords_y
            self.coords(self.h_line, 0, h_line_y, canvas_w, h_line_y)


            img_x, img_y = self._get_image_coords_from_canvas(coords_x, coords_y)
            if not (0 <= img_y < self.capture_hsi.shape[0] and 0 <= img_x < self.capture_hsi.shape[1]):
                return
            if self.data_plots:
                self.data_plots['reflectance'].update_dynamic_line(self.capture_hsi[img_y, img_x], name=self.plot_names['reflectance'])
                if self.raw_hsi is not None:
                    self.data_plots['raw'].update_dynamic_line(self.raw_hsi[img_y, img_x], name=self.plot_names['raw'])
                else:
                    self.data_plots['raw'].clear_dynamic_line(self.plot_names['raw'])
            else:
                self.status["text"] = "No Detection Data " + self.status["text"]
        except Exception as e:
            pass

    def set_input_data(self, capture_hsi, capture_rgb, raw_hsi=None):
        self.capture_hsi = capture_hsi
        self.raw_hsi = raw_hsi
        self.capture_image = capture_rgb
        self._update_gt_mean_spectra()
        self.move_lines(self.v_line_x, self.h_line_y)
        self.update_image_view(img_rgb=self.capture_image)

    def update_image_view(self, img_rgb=None, img_bgr=None):
        assert(img_rgb is None or img_bgr is None) or (img_rgb is None and img_bgr is None), "Provide either rgb or bgr argument"

        if img_rgb is not None:
            if self.show_gt and self.gt_rgb is not None and img_rgb.shape == self.gt_rgb.shape:
                img_to_show = cv2.addWeighted(img_rgb, 0.6, self.gt_rgb, 0.4, 0)
            else:
                img_to_show = img_rgb
            img = Image.fromarray(img_to_show)
        else:
            try:
                blue,green,red = cv2.split(img_bgr)
                capture_img_rgb = cv2.merge((red,green,blue))
                if self.show_gt and self.gt_rgb is not None and capture_img_rgb.shape == self.gt_rgb.shape:
                    capture_img_rgb = cv2.addWeighted(capture_img_rgb, 0.6, self.gt_rgb, 0.4, 0)
                img = Image.fromarray(capture_img_rgb)
            except:
                img = img_bgr

        canvas_w = self.winfo_width() if self.winfo_width() > 10 else int(self["width"])
        canvas_h = self.winfo_height() if self.winfo_height() > 10 else img.height
        resized_image = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(image=resized_image)
        self.itemconfig(self.canvas_image, image=self.tk_image)

        self.status["text"] = f'({img.height},{img.width})'
        self.status.place(x=(self.winfo_x() + canvas_w), y=(self.winfo_y() + canvas_h + 3), anchor='ne')


    def overlap_images(self, img_1, img_2):
        self.overlap_result = cv2.addWeighted(img_1, 0.8, img_2, 0.5, 0)
        self.update_image_view(img_rgb=self.overlap_result)

    def get_selected_rgb_indices(self):
        return self.data_plots['reflectance'].get_closest_selected_rgb_indices(self.plot_names['reflectance'])

    def _update_gt_mean_spectra(self):
        """Show static means for every annotated pixel in the current sample."""
        ref_mean = compute_mean_gt_spectrum(self.capture_hsi, self.gt_mask)
        raw_mean = compute_mean_gt_spectrum(self.raw_hsi, self.gt_mask) if self.raw_hsi is not None else None
        self.data_plots['reflectance'].update_mean_gt_line(ref_mean, self.plot_names['reflectance'], class_name='Annotated sample')
        self.data_plots['raw'].update_mean_gt_line(raw_mean, self.plot_names['raw'], class_name='Annotated sample')



class ClassificationTable(ttk.Treeview):
    def __init__(self, root):
        super().__init__(root, height=1, selectmode='none')
        self.column("#0", width=0,  stretch=NO)
        self.heading("#0", text="", anchor=CENTER)
        columns = ()
        for id,cat in read_categories().items():
            columns += (('#'+str(id)),)
        
        self['columns'] = columns

        values = ()

        for id,cat in read_categories().items(): 
            self.column(('#'+str(id)), anchor=CENTER, width=110)
            self.heading(('#'+str(id)), text=(str(id) + ': ' + cat), anchor=CENTER)
            values += ('',)

        self.insert(parent='', index='end', iid=0, text='NPY GT', values=values)

        sel_bg = '#ecffc4'
        sel_fg = '#05640e'
        self.setup_selection(sel_bg, sel_fg)
        sel_bg = '#FFE4B5'
        self.setup_selection_2nd(sel_bg, sel_fg)

        self.hover_id = None      
        self.popup = None         
        self.last_cell = None     


    def setup_selection(self, sel_bg, sel_fg):
        self._font = tkFont.Font()

        self._canvas = tk.Canvas(self,
                                 background=sel_bg,
                                 borderwidth=0,
                                 highlightthickness=0)

        self._canvas.text = self._canvas.create_text(0, 0,
                                                     fill=sel_fg,
                                                     anchor='w')    

    def cell_highlight(self, column, row=0):
        """Configure canvas and canvas-textbox for a new selection."""
        if column == -1:
            self._canvas.place_forget()
            return
        x, y, width, height = self.bbox(row, column)
        fudgeTreeColumnx = 0 
        fudgeColumnx = 2    

        row_items = row
        textw = self._font.measure(self.item(row_items, 'values')[column])

        self._canvas.configure(width=width, height=height)

        self._canvas.coords(self._canvas.text,
                            (width-(textw-fudgeColumnx))/2.0,
                            height/2)

        self._canvas.itemconfigure(self._canvas.text, text=self.item(row_items, 'values')[column])

        self._canvas.place(in_=self, x=x, y=y)

    def setup_selection_2nd(self, sel_bg, sel_fg):
        self._font = tkFont.Font()

        self._canvas2nd = tk.Canvas(self,
                                 background=sel_bg,
                                 borderwidth=0,
                                 highlightthickness=0)

        self._canvas2nd.text = self._canvas2nd.create_text(0, 0,
                                                     fill=sel_fg,
                                                     anchor='w')    
        self._secondary_canvases = [self._canvas2nd]

    def _create_secondary_canvas(self):
        canvas = tk.Canvas(self, background='#FFE4B5', borderwidth=0, highlightthickness=0)
        canvas.text = canvas.create_text(0, 0, fill='#05640e', anchor='w')
        self._secondary_canvases.append(canvas)
        return canvas

    def _place_highlight(self, canvas, column, row):
        x, y, width, height = self.bbox(row, column)
        value = self.item(row, 'values')[column]
        textw = self._font.measure(value)
        canvas.configure(width=width, height=height)
        canvas.coords(canvas.text, (width - (textw - 2)) / 2.0, height / 2)
        canvas.itemconfigure(canvas.text, text=value)
        canvas.place(in_=self, x=x, y=y)

    def cell_highlight_2nd(self, column, row=0):
        if column == -1:
            for canvas in self._secondary_canvases:
                canvas.place_forget()
            return
        self._place_highlight(self._canvas2nd, column, row)

    def highlight_secondary_columns(self, columns, row=0):
        """Highlight every additional positive annotation in orange."""
        for index, column in enumerate(columns):
            canvas = self._secondary_canvases[index] if index < len(self._secondary_canvases) else self._create_secondary_canvas()
            self._place_highlight(canvas, int(column), row)
        for canvas in self._secondary_canvases[len(columns):]:
            canvas.place_forget()

    def reset_data(self):
        values = ()
        for _ in range(len(self['columns'])): values += "",
        self.item(0, values=values)
        self.cell_highlight(-1)
        self.cell_highlight_2nd(-1)

    def insert_data(self, values, as_percentage=False):
        values = np.asarray(values, dtype=float)
        display_values = values * 100 if as_percentage else values
        rounded_values = np.round(display_values, 1 if as_percentage else 2)
        self.item(0, values=tuple(rounded_values))
        prds = np.argsort(rounded_values, axis=0)
        if np.all(rounded_values <= 0):
            self.cell_highlight(-1)
            self.cell_highlight_2nd(-1)
        else:
           
            self.cell_highlight(int(prds[-1]), row=0)
            secondary_columns = [int(prds[-i]) for i in range(2, len(rounded_values) + 1)
                                 if rounded_values[prds[-i]] > 0.0]
            self.highlight_secondary_columns(secondary_columns, row=0)

    def insert_gt_data(self, values):
        """Backward-compatible alias for the NPY GT row update."""
        if values is not None:
            self.insert_data(values, as_percentage=True)


class HSIPlot(tk.Frame):
    class DraggableVerticalLine:
        def __init__(self, ax, x, color='red', x_min=None, x_max=None):
            self.ax = ax
            self.color = color
            self.x_min = x_min
            self.x_max = x_max

            # Create the vertical line
            self.line = Line2D([x, x], ax.get_ylim(), color=color, linestyle='--', lw=0.5)
            self.ax.add_line(self.line)

            # Create the text label
            self.label = self.ax.text(
                x, -0.1, f'{x:.1f}', color=color,
                ha='center', va='top', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                transform=self.ax.get_xaxis_transform()
            )

            self.press = None
            self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
            self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
            self.cid_motion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        def on_press(self, event):
            if event.inaxes != self.ax:
                return
            contains, _ = self.line.contains(event)
            if contains:
                self.press = event.xdata

        def on_release(self, event):
            self.press = None

        def on_motion(self, event):
            if self.press is None or event.inaxes != self.ax:
                return

            x = event.xdata
            if self.x_min is not None:
                x = max(x, self.x_min)
            if self.x_max is not None:
                x = min(x, self.x_max)

            self.line.set_xdata([x, x])

            self.label.set_position((x, -0.1))
            self.label.set_text(f'{x:.1f}')

            self.line.figure.canvas.draw_idle()

        def get_x(self):
            return self.line.get_xdata()[0]
        
    def __init__(self, root, title="Spectral Data", ylabel="Reflectance", ylim=None):
        super().__init__(root)
        self.root = root
        self.bands = {}
        self.dynamic_lines = {}
        self.gt_mean_lines = {}
        self.counter = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 3.9))
        self.fig.set_layout_engine('tight')

        self.ax.set_xlim(800, 1800)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_title(title)
        self.ax.grid(True)

        self.implot = FigureCanvasTkAgg(self.fig, master=self)
        self.implot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.draggable_line_sets = {}

    def add_bands_configuration(self, bands, color='g-'):
        dynamic_line, = self.ax.plot([], [], color, linewidth=2, label="Current Pixel")
        name = f'#{self.counter}'
        self.counter +=1
        self.dynamic_lines[name] = dynamic_line
        self.bands[name] = bands
        return name

    def clear_plot(self, name):
        for line in self.ax.get_lines(): 
            if line.get_label() == name: line.remove()
        
        self.dynamic_lines[name].set_data([], [])
        self.fig.canvas.draw_idle() 
    
    def update_dynamic_line(self, hsi_data, name):
        if name not in self.dynamic_lines: return 
        self.dynamic_lines[name].set_data(self.bands[name], hsi_data)
        self.fig.canvas.draw_idle()

    def clear_dynamic_line(self, name):
        if name in self.dynamic_lines:
            self.dynamic_lines[name].set_data([], [])
            self.fig.canvas.draw_idle()

    def update_mean_gt_line(self, mean_gt_spectrum, name, class_name="GT Region"):
        if name not in self.bands:
            return
        if mean_gt_spectrum is None:
            if name in self.gt_mean_lines:
                self.gt_mean_lines[name].set_data([], [])
                self.fig.canvas.draw_idle()
            return

        lbl = f"Mean GT: {class_name}"
        if name not in self.gt_mean_lines:
            line, = self.ax.plot(self.bands[name], mean_gt_spectrum, 'g--', linewidth=2, label=lbl)
            self.gt_mean_lines[name] = line
        else:
            self.gt_mean_lines[name].set_data(self.bands[name], mean_gt_spectrum)
            self.gt_mean_lines[name].set_label(lbl)
        
        self.ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()

    def set_plot_data(self, hsi_data, name, color=None):
        line, = self.ax.plot(self.bands[name], hsi_data, label=name, color=color)
        self.implot.draw()
        return line.get_color()
    
    def add_draggable_lines(self, min_x, max_x, name, colors=None, rgb_bands=None):
        """
        Adds 3 draggable vertical lines bounded to [min_x, max_x].
        """
        if colors is None:
            colors = ['red', 'green', 'blue']

        line_set = []
        for i in range(3):
            line = self.DraggableVerticalLine(self.ax, self.bands[name][rgb_bands[i]], color=colors[i % len(colors)],
                                              x_min=min_x, x_max=max_x)
            line_set.append(line)

        self.draggable_line_sets[name] = line_set
        self.implot.draw()

    def get_line_positions(self, name):
        """
        Returns a dictionary: {(min_x, max_x): [x1, x2, x3]}
        """
        return [line.get_x() for line in self.draggable_line_sets[name]]

    def get_closest_selected_rgb_indices(self, name):
        rgb_bands = self.get_line_positions(name)
        rgb_indices = np.abs(np.asarray(self.bands[name])[None, :] - np.asarray(rgb_bands)[:, None]).argmin(axis=1)
        return rgb_indices


class WinProgressBar(tk.Frame):
    def __init__(self, master=None, place=None, length=200):
        super().__init__(master)
        self.master = master
        self.progressbar = ttk.Progressbar(self.master, orient=tk.HORIZONTAL, mode='indeterminate', length=length)
        self.progressbar.place(x=place[0], y=place[1])
        self.progressbar.start(15)


if __name__ == "__main__":
    gui         = SpecGUI()
    gui.start()
