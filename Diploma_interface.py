from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from Diploma_OCR_classes import CharacterDetector, Classifier

character_dict = {0: '一', 1: '七', 2: '三', 3: '上', 4: '下', 5: '不', 6: '东', 7: '么', 8: '九', 9: '习', 10: '书', 11: '买', 12: '了', 
                  13: '二', 14: '五', 15: '些', 16: '亮', 17: '人', 18: '什', 19: '今', 20: '他', 21: '们', 22: '会', 23: '住', 24: '作', 
                  25: '你', 26: '候', 27: '做', 28: '儿', 29: '先', 30: '八', 31: '六', 32: '关', 33: '兴', 34: '再', 35: '写', 36: '冷', 
                  37: '几', 38: '出', 39: '分', 40: '前', 41: '北', 42: '医', 43: '十', 44: '午', 45: '去', 46: '友', 47: '吃', 48: '同', 
                  49: '名', 50: '后', 51: '吗', 52: '呢', 53: '和', 54: '哪', 55: '商', 56: '喂', 57: '喜', 58: '喝', 59: '四', 60: '回', 
                  61: '国', 62: '在', 63: '坐', 64: '块', 65: '多', 66: '大', 67: '天', 68: '太', 69: '她', 70: '好', 71: '妈', 72: '姐', 
                  73: '子', 74: '字', 75: '学', 76: '客', 77: '家', 78: '对', 79: '小', 80: '少', 81: '岁', 82: '工', 83: '师', 84: '年', 
                  85: '店', 86: '开', 87: '影', 88: '很', 89: '怎', 90: '想', 91: '我', 92: '打', 93: '日', 94: '时', 95: '明', 96: '星', 
                  97: '昨', 98: '是', 99: '月', 100: '有', 101: '朋', 102: '服', 103: '期', 104: '本', 105: '机', 106: '来', 107: '杯', 
                  108: '果', 109: '校', 110: '样', 111: '桌', 112: '椅', 113: '欢', 114: '气', 115: '水', 116: '汉', 117: '没', 118: '漂', 
                  119: '火', 120: '点', 121: '热', 122: '爱', 123: '爸', 124: '狗', 125: '猫', 126: '现', 127: '生', 128: '电', 129: '的',
                  130: '看', 131: '睡', 132: '租', 133: '站', 134: '米', 135: '系', 136: '老', 137: '能', 138: '脑', 139: '苹', 140: '茶', 
                  141: '菜', 142: '衣', 143: '西', 144: '见', 145: '视', 146: '觉', 147: '认', 148: '识', 149: '话', 150: '语', 151: '说', 
                  152: '请', 153: '读', 154: '谁', 155: '谢', 156: '起', 157: '车', 158: '这', 159: '那', 160: '都', 161: '里', 162: '钟', 
                  163: '钱', 164: '院', 165: '雨', 166: '零', 167: '面', 168: '飞', 169: '饭', 170: '馆', 171: '高'}

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text   = text
        self.tooltip_window = None

        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event = None):
        if self.tooltip_window:
            return
        
        x, y, _, _ = self.widget.bbox("insert")

        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(self.tooltip_window, text = self.text, justify = 'left', background = 'lightyellow', relief = 'solid', borderwidth = 1, font = ("arial", "10", "normal"))
        label.pack(ipadx = 1)

    def hide_tooltip(self, event = None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class MainWindow(Tk):
    def __init__(self):
        super().__init__()
        self.title("Головне меню")
        self.geometry(f"570x360")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 570) // 2
        y = (screen_height - 360) // 2
        self.geometry(f"570x360+{x}+{y}")
        self.resizable(width = False, height = False)

        self.iconphoto(False, PhotoImage(file = "E:/Work Folder/0_Polya/ДТЕУ/Дипломна робота/Diploma/interface_icon.jpg"))

        main_frame = ttk.Frame(self)
        main_frame.pack(expand = True, pady = 50)

        style = ttk.Style()
        style.configure("TButton", font = ("Arial", 16), padding = (10, 20))

        # Buttron for recognizing one character
        btn_single = ttk.Button(main_frame, text = "Розпізнати один китайський ієрогліф", command = self.open_single_char_window, style = 'TButton')
        btn_single.pack(pady = 30)

        # Button for text recognition
        btn_text = ttk.Button(main_frame, text = "Розпізнати китайський текст", command = self.open_text_recognition, style = 'TButton')
        btn_text.pack(pady = 30)

        self.text_window   = None
        self.single_window = None

    def open_single_char_window(self):
        # Close main window
        self.withdraw()

        # Open window for single character recognition
        self.single_window = Toplevel()
        self.single_window.title("Розпізнавання одного ієрогліфа")
        self.single_window.geometry("600x400")
        self.single_window.resizable(width = False, height = False)
        self.single_window.iconphoto(False, PhotoImage(file = "E:/Work Folder/0_Polya/ДТЕУ/Дипломна робота/Diploma/interface_icon.jpg"))

        single_char_app = SingleCharApp(self.single_window)

        # Add protocol to handle window closing
        self.single_window.protocol("WM_DELETE_WINDOW", self.on_single_window_close)

    def open_text_recognition(self):
        # Close main window
        self.withdraw()

        # Open window for text recognition
        self.text_window = Toplevel()
        text_app = App(self.text_window)
        #text_app.master.protocol("WM_DELETE_WINDOW", lambda: self.destroy_and_show_main())
        self.text_window.protocol("WM_DELETE_WINDOW", self.on_text_window_close)

    def on_single_window_close(self):
        # If window for single character recognition is closing, return to main window
        self.single_window.destroy()
        self.single_window = None
        
        style = ttk.Style()
        style.configure("TButton", font = ("Arial", 16), padding = (10, 20))
        
        self.deiconify()
    
    def on_text_window_close(self):
        # If window for text recognition is closing, return to main window
        self.text_window.destroy()
        self.text_window = None

        style = ttk.Style()
        style.configure("TButton", font = ("Arial", 16), padding = (10, 20))

        self.deiconify()

class App(Frame):
    def __init__(self, master):
        super().__init__(master)

        # Setting up the main application window
        self.setup_main_window(master)

        # Variable to store information
        self.H_MARGIN_ERROR      = StringVar()
        self.W_MARGIN_ERROR      = StringVar()
        self.GAUSSIANBLUR_SIGMAX = StringVar()
        self.GAUSSIANBLUR_KSIZE  = StringVar()
        self.MORPH_KSIZE         = StringVar()

        # Create interface elements
        self.create_frames(master)
        self.create_canvases()
        self.create_buttons()
        self.create_entries()
        self.create_prep_img_checkbox()
        self.create_bounding_boxes_checkbox()
        self.create_recodnized_chars_checkbox()
        self.create_labels()       

        # Customizing the widgets layout
        self.arrange_widgets()                     

    def setup_main_window(self, master):
        master.title("Система розпізнавання китайського тексту")
        master.geometry(f"1140x720+198+37")
        #master.resizable(width = False, height = False)
        master.resizable(width = True, height = True)
        master.iconphoto(False, PhotoImage(file = "E:/Work Folder/0_Polya/ДТЕУ/Дипломна робота/Diploma/interface_icon.jpg"))

    def get_frame_size(self, frame):
        width  = frame.winfo_width()
        height = frame.winfo_height()

        return width, height

    def create_frames(self, master):
        self.text_photo_frame                = ttk.Frame(master, width = 657, height = 360)
        self.recog_frame                     = ttk.Frame(master, width = 657, height = 360)
        self.button_frame                    = ttk.Frame(master)
        self.show_prep_container             = ttk.Frame(self.button_frame)
        self.show_bounding_boxes_container   = ttk.Frame(self.button_frame)
        self.show_recodnized_chars_container = ttk.Frame(self.button_frame)
        self.h_margin_er_frame               = ttk.Frame(self.button_frame)
        self.w_margin_er_frame               = ttk.Frame(self.button_frame)
        self.GaussianBlur_sigmaX_frame       = ttk.Frame(self.button_frame)
        self.GaussianBlur_ksize_frame        = ttk.Frame(self.button_frame)
        self.morph_ksize_frame               = ttk.Frame(self.button_frame)
        self.answer_frame                    = ttk.Frame(self.button_frame)
        self.text_photo_frame.grid_propagate(False)
        self.recog_frame.grid_propagate(False)  
        self.button_frame.grid_propagate(False)

    def create_canvases(self):
        self.text_photo_canvas = Canvas(self.text_photo_frame, width = 657, height = 360, background = 'gray85', highlightbackground = "black")
        self.recognized_canvas = Canvas(self.recog_frame, width = 657, height = 360, background = 'gray85', highlightbackground = "black")
        self.answer_canvas     = Canvas(self.answer_frame, background = 'gray85', highlightbackground = "black")

    def create_buttons(self):
        self.load_picture_button = ttk.Button(self.button_frame, text = 'Завантажити зображення', width = 25, command = self.load_image)
        self.recognize_button    = ttk.Button(self.button_frame, text = 'Розпізнати', width = 18, command = self.recognize_text)
        ttk.Style().configure("TButton", padding = (0, 5, 0, 5), font = 'arial 14') # Configure button styles

    def create_entries(self):
        self.h_margin_er_entry         = ttk.Entry(self.h_margin_er_frame, textvariable = self.H_MARGIN_ERROR, font = ("default", (14)), width = 5)
        self.w_margin_er_entry         = ttk.Entry(self.w_margin_er_frame, textvariable = self.W_MARGIN_ERROR, font = ("default", (14)), width = 5)
        self.GaussianBlur_sigmaX_entry = ttk.Entry(self.GaussianBlur_sigmaX_frame, textvariable = self.GAUSSIANBLUR_SIGMAX, font = ("default", (14)), width = 5)
        self.GaussianBlur_ksize_entry  = ttk.Entry(self.GaussianBlur_ksize_frame, textvariable = self.GAUSSIANBLUR_KSIZE, font = ("default", (14)), width = 5)
        self.morph_ksize_entry         = ttk.Entry(self.morph_ksize_frame, textvariable = self.MORPH_KSIZE, font = ("default", (14)), width = 5)

    def create_labels(self):
        self.show_prep_label             = Label(self.show_prep_container, text = "Показати препроцесинг зображення:", font = 'arial 14')
        self.show_bounding_boxes_label   = Label(self.show_bounding_boxes_container, text = "Показати контури ієрогліфів:", font = 'arial 14')
        self.show_recodnized_chars_label = Label(self.show_recodnized_chars_container, text = "Показати розпізнані ієрогліфи:", font = 'arial 14')
        self.w_margin_er_label           = Label(self.w_margin_er_frame, text = 'Відсоток похибки ширини контура (0-1):', font = 'arial 14')
        self.h_margin_er_label           = Label(self.h_margin_er_frame, text = 'Відсоток похибки висоти контура (0-1):', font = 'arial 14')
        self.GaussianBlur_sigmaX_label   = Label(self.GaussianBlur_sigmaX_frame, text = 'Інтенсивність розмиття по X:', font = 'arial 14')
        self.GaussianBlur_ksize_label    = Label(self.GaussianBlur_ksize_frame, text = 'Розмір ядра розмиття:', font = 'arial 14')
        self.morph_ksize_label           = Label(self.morph_ksize_frame, text = 'Розмір фільтра морфологічних операцій:', font = 'arial 14')

        # Add "ⓘ" labels for tooltips
        self.w_margin_er_info         = Label(self.w_margin_er_frame, text = "ⓘ", font = 'arial 14', foreground = "blue", cursor = "hand2")
        self.h_margin_er_info         = Label(self.h_margin_er_frame, text = "ⓘ", font = 'arial 14', foreground = "blue", cursor = "hand2")
        self.GaussianBlur_sigmaX_info = Label(self.GaussianBlur_sigmaX_frame, text = "ⓘ", font = 'arial 14', foreground = "blue", cursor = "hand2")
        self.GaussianBlur_ksize_info  = Label(self.GaussianBlur_ksize_frame, text = "ⓘ", font = 'arial 14', foreground = "blue", cursor = "hand2")
        self.morph_ksize_info         = Label(self.morph_ksize_frame, text = "ⓘ", font = 'arial 14', foreground = "blue", cursor = "hand2")

        # Attach tooltips
        ToolTip(self.w_margin_er_info, "Відсоток допустимої похибки\nдля ширини контура.\nПриймає значення від 0 до 1.")
        ToolTip(self.h_margin_er_info, "Відсоток допустимої похибки\nдля висоти контура.\nПриймає значення від 0 до 1.")
        ToolTip(self.GaussianBlur_sigmaX_info, "Задає стандартне відхилення розподілу\nГаусовського по осі X для фільтра розмиття.\nПараметр приймає додатні числа\n(включаючи десяткові).\nЩо більше значення параметра,\nто сильніше розмиття.\nЯкщо параметр = 0, стандартне відхилення\nбуде обчислено автоматично на основі\nрозміру ядра (параметр ksize).")
        ToolTip(self.GaussianBlur_ksize_info, "Задає розмір ядра (kernel size), що використовується\nдля Гаусовського розмиття.\nЦе визначає, наскільки велике вікно буде\nвикористовуватися для обчислення розмиття.\nПараметр ksize є кортежем з двох значень\n(width, height).\nРозміри ядра мають бути непарними числами\n(наприклад, (5, 5), (1, 3)).")
        ToolTip(self.morph_ksize_info, "Вказує розмір ядра для\nморфологічної операції.\nЦей параметр задається як\nкортеж із двох чисел\n(height, width) для прямокутного\nчи квадратного ядра\n(наприклад, (2, 2)).")

    def create_prep_img_checkbox(self):
        # Create style for Checkbutton
        style = ttk.Style()
        style.configure('Custom.TCheckbutton', font = 'arial 14', foreground = 'black')

        self.show_prep_var      = BooleanVar()
        self.show_prep_checkbox = ttk.Checkbutton(self.show_prep_container, variable = self.show_prep_var, onvalue = True, offvalue = False, 
                                                  text = 'так', style = 'Custom.TCheckbutton')

    def create_bounding_boxes_checkbox(self):
        # Create style for Checkbutton
        style = ttk.Style()
        style.configure('Custom.TCheckbutton', font = 'arial 14', foreground = 'black')

        self.show_bounding_boxes_var      = BooleanVar()
        self.show_bounding_boxes_checkbox = ttk.Checkbutton(self.show_bounding_boxes_container, variable = self.show_bounding_boxes_var, 
                                                            onvalue = True, offvalue = False, 
                                                            text = 'так', style = 'Custom.TCheckbutton')
        
    def create_recodnized_chars_checkbox(self):
        # Create style for Checkbutton
        style = ttk.Style()
        style.configure('Custom.TCheckbutton', font = 'arial 14', foreground = 'black')

        self.show_recodnized_chars_var      = BooleanVar()
        self.show_recodnized_chars_checkbox = ttk.Checkbutton(self.show_recodnized_chars_container, variable = self.show_recodnized_chars_var, 
                                                              onvalue = True, offvalue = False, 
                                                              text = 'так', style = 'Custom.TCheckbutton')

    def arrange_widgets(self):
        # Setting up scales
        self.master.grid_columnconfigure(0, weight = 1)
        self.master.grid_columnconfigure(1, weight = 1)
        self.master.grid_columnconfigure(2, weight = 1)
        self.master.grid_columnconfigure(3, weight = 1)

        self.master.grid_rowconfigure(0, weight = 1)
        self.master.grid_rowconfigure(1, weight = 1)
        self.master.grid_rowconfigure(2, weight = 1)

        # Frame for photo with text
        self.text_photo_frame.grid(row = 0, column = 0, columnspan = 3, sticky = "nsew")
        self.text_photo_canvas.pack(fill = 'both', expand = True) # Set canvas sizes equal to frame sizes initially
        self.text_photo_canvas.create_text(328, 180,
                                           text = 'Для завантаження зображення з текстом\nнатисніть кнопку\n"Завантажити зображення"',
                                           font = ("Arial", 14),
                                           fill = "black",
                                           justify = "center")

        # Frame for bounding boxes and recognition results
        self.recog_frame.grid(row = 1, column = 0, columnspan = 3, sticky = "nsew")
        self.recognized_canvas.pack(fill = 'both', expand = True) # Set canvas sizes equal to frame sizes initially
        self.recognized_canvas.create_text(328, 180,
                                           text = 'Для розпізнання тексту на зображенні\nнатисніть кнопку\n"Розпізнати"',
                                           font = ("Arial", 14),
                                           fill = "black",
                                           justify = "center")

        # Button frame
        self.button_frame.grid(row = 0, rowspan = 2, column = 3, sticky = "nsew")
        self.load_picture_button.pack(pady = 5, anchor = "center")

        self.show_prep_label.pack(side = LEFT, padx = 5, pady = 5)
        self.show_prep_checkbox.pack(side = LEFT, padx = 5, pady = 5)
        self.show_prep_container.pack(padx = 10, pady = 5, anchor = "w")

        self.show_bounding_boxes_label.pack(side = LEFT, padx = 5, pady = 5)
        self.show_bounding_boxes_checkbox.pack(side = LEFT, padx = 5, pady = 5)
        self.show_bounding_boxes_container.pack(padx = 10, pady = 5, anchor = "w")

        self.show_recodnized_chars_label.pack(side = LEFT, padx = 5, pady = 5)
        self.show_recodnized_chars_checkbox.pack(side = LEFT, padx = 5, pady = 5)
        self.show_recodnized_chars_container.pack(padx = 10, pady = 5, anchor = "w")
        
        self.h_margin_er_label.pack(side = LEFT, padx = 5, pady = 5)
        self.h_margin_er_entry.pack(side = LEFT, padx = 5, pady = 5)
        self.h_margin_er_info.pack(side = LEFT, padx = 5)
        self.h_margin_er_frame.pack(padx = 10, pady = 5, anchor = "w")
        
        self.w_margin_er_label.pack(side = LEFT, padx = 5, pady = 5)
        self.w_margin_er_entry.pack(side = LEFT, padx = 5, pady = 5)
        self.w_margin_er_info.pack(side = LEFT, padx = 5)
        self.w_margin_er_frame.pack(padx = 10, pady = 5, anchor = "w")

        self.GaussianBlur_sigmaX_label.pack(side = LEFT, padx = 5, pady = 5)
        self.GaussianBlur_sigmaX_entry.pack(side = LEFT, padx = 5, pady = 5)
        self.GaussianBlur_sigmaX_info.pack(side = LEFT, padx = 5)
        self.GaussianBlur_sigmaX_frame.pack(padx = 10, pady = 5, anchor = "w")

        self.GaussianBlur_ksize_label.pack(side = LEFT, padx = 5, pady = 5)
        self.GaussianBlur_ksize_entry.pack(side = LEFT, padx = 5, pady = 5)
        self.GaussianBlur_ksize_info.pack(side = LEFT, padx = 5)
        self.GaussianBlur_ksize_frame.pack(padx = 10, pady = 5, anchor = "w")

        self.morph_ksize_label.pack(side = LEFT, padx = 5, pady = 5)
        self.morph_ksize_entry.pack(side = LEFT, padx = 5, pady = 5)
        self.morph_ksize_info.pack(side = LEFT, padx = 5)
        self.morph_ksize_frame.pack(padx = 10, pady = 5, anchor = "w")

        self.recognize_button.pack(pady = 5, anchor = "center")

        # Answer frame
        self.answer_canvas.pack(fill = 'both', expand = True) # Set canvas sizes equal to frame sizes initially
        self.answer_frame.pack(pady = 5, anchor = "s")

    def load_image(self):
        # Clear answer_canvas 
        self.answer_canvas.delete("all")

        # Clear recognized_canvas and add text
        self.recognized_canvas.delete("all")
        self.recognized_canvas.create_text(328, 180,
                                           text = 'Для розпізнання тексту на зображенні\nнатисніть кнопку\n"Розпізнати"',
                                           font = ("Arial", 14),
                                           fill = "black",
                                           justify = "center")

        self.filename = filedialog.askopenfilename() # Open file with image
        if not self.filename:
            return
        
        image = Image.open(self.filename)

        self.show_image(image, 'load')

    def recognize_text(self):
        # Values for the parameters
        show_bounding_boxes   = self.show_bounding_boxes_var.get()
        show_recodnized_chars = self.show_recodnized_chars_var.get()
        h_margin_error        = float(self.h_margin_er_entry.get())
        w_margin_error        = float(self.w_margin_er_entry.get())
        GaussianBlur_sigmaX   = int(self.GaussianBlur_sigmaX_entry.get())
        GaussianBlur_ksize    = eval(self.GaussianBlur_ksize_entry.get())
        morph_ksize           = eval(self.morph_ksize_entry.get())
        show_prep_img         = self.show_prep_var.get()

        model_path = 'model_improved.keras'

        # Load image
        pil_image = Image.open(self.filename)
        image     = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        character_detector = CharacterDetector(show_bounding_boxes, h_margin_error, w_margin_error, GaussianBlur_ksize, GaussianBlur_sigmaX, morph_ksize)

        morph = character_detector.preprocess_img(image, show_prep_img)

        detected_img, resulted_contours = character_detector.detect_contours(image, morph)

        classifier = Classifier(model_path, detected_img, resulted_contours, character_dict, image, show_recodnized_chars)
        recognized_img, resulted_chars, median_acc = classifier.recognize_characters()

        final_image = Image.fromarray(cv2.cvtColor(recognized_img, cv2.COLOR_BGR2RGB))
        self.show_image(final_image, 'recognize')

        self.show_text(resulted_chars, median_acc)
    
    def show_image(self, image_to_show, action: str):
        if action == 'load':
            frame     = self.text_photo_frame
            canvas    = self.text_photo_canvas
            attribute = "loaded_image"

        elif action == 'recognize':
            frame     = self.recog_frame
            canvas    = self.recognized_canvas
            attribute = "recognized_image"

        else:
            raise ValueError(f"Unknown action: {action}")

        # Get frame's size
        frame_width, frame_height = self.get_frame_size(frame)

        # Resize the image to fit the frame
        max_size = (frame_width, frame_height)
        image_to_show.thumbnail(max_size)  # Scale the image proportionally

        # Set the canvas size to the size of the resized image
        img_width, img_height = image_to_show.size
        canvas.config(width = frame_width, height = frame_height)

        # Create a PhotoImage object and save it as an attribute
        setattr(self, attribute, ImageTk.PhotoImage(image_to_show))

        # Calculate coordinates for image centering
        x_offset = (frame_width - img_width) // 2
        y_offset = (frame_height - img_height) // 2

        # Clear the canvas 
        canvas.delete("all")

        # Display image on canvas with center alignment
        canvas.create_image(x_offset, y_offset, anchor = "nw", image = getattr(self, attribute))

    def show_text(self, resulted_chars, accuracy):
        # Combine each inner list into a single string
        text_lines = [''.join(line) for line in resulted_chars]

        # Set initial y-coordinate for the text
        y = 20
        line_spacing = 20  # Space between lines

        for line in text_lines:
            self.answer_canvas.create_text(10, y, anchor = "w", text = line, font = ("Arial", 14))

            y += line_spacing

        # Get the height and width of the canvas
        canvas_width  = self.answer_canvas.winfo_width()
        canvas_height = self.answer_canvas.winfo_height()

        # Round accuracy and format it as a string with three decimal places
        accuracy_str = f"{accuracy:.3f}"  # Format to 3 decimal places

        # Add accuracy at the bottom center
        self.answer_canvas.create_text(canvas_width / 2, canvas_height - 10, 
                                       anchor = "s", 
                                       text = f"Точність (медіанне значення) = {accuracy_str}", 
                                       font = ("Arial", 14)
                                       )

class SingleCharApp(Frame):
    def __init__(self, master):
        super().__init__(master)

        self.IS_ERASE = False
        self.model_path = 'model_improved.keras'  # Same model as in text recognition

        self.image_numpy = np.zeros(shape=(64, 64), dtype=np.uint8)  # Using 64x64 for consistency

        master.title("Розпізнавання одного ієрогліфа")
        master.geometry("570x360+475+150")
        master.resizable(width = False, height = False)
        master.iconphoto(False, PhotoImage(file = "E:/Work Folder/0_Polya/ДТЕУ/Дипломна робота/Diploma/interface_icon.jpg"))

        canvas_frame     = ttk.Frame(master)
        button_frame     = ttk.Frame(master)
        answer_frame     = ttk.Frame(button_frame)
        self.result      = StringVar()
        self.canvas      = Canvas(canvas_frame, width = 64 * 5, height = 64 * 5, background = 'gray74')  # Scaled up by 5
        draw_button      = ttk.Button(button_frame, text = 'Написати', width = 18, command = self.draw_mode)
        erase_button     = ttk.Button(button_frame, text = 'Стерти', width = 18, command = self.erase_mode)
        clear_button     = ttk.Button(button_frame, text = 'Очистити екран', width = 18, command = self.clear_screen)
        load_button      = ttk.Button(button_frame, text = 'Завантажити картинку', width = 18, command = self.load_picture)
        recognize_button = ttk.Button(button_frame, text = 'Розпізнати', width = 18, command = self.recognize_char)
        label_result     = Label(answer_frame, textvariable = self.result, font = 'arial 14', background = 'gray64', width = 15, height = 2)

        ttk.Style().configure("TButton", padding = (0, 5, 0, 5), font = 'arial 12')

        canvas_frame.grid(row = 0, column = 0, padx = 10, pady = 15)
        self.canvas.pack(side = LEFT)
        button_frame.grid(row = 0, column = 1)
        draw_button.pack(padx = 25, pady = 10)
        erase_button.pack(padx = 25, pady = 10)
        clear_button.pack(padx = 25, pady = 10)
        load_button.pack(padx = 25, pady = 10)
        recognize_button.pack(padx = 25, pady = 10)
        answer_frame.pack(padx = 25, pady = 10)
        label_result.grid(column = 0, row = 0)

        self.canvas_squares = []
        for y in range(64):
            row_squares = []
            for x in range(64):
                px = x * 5
                py = y * 5
                square = self.canvas.create_rectangle(px, py, px + 5, py + 5, fill = 'black')
                row_squares.append(square)
            self.canvas_squares.append(row_squares)

        self.canvas.bind('<ButtonPress-1>', self.canvas_mouse_down)
        self.canvas.bind('<B1-Motion>', self.canvas_mouse_move)

    def draw_mode(self):
        self.IS_ERASE = False

    def erase_mode(self):
        self.IS_ERASE = True

    def clear_screen(self):
        for y in range(64):
            for x in range(64):
                square = self.canvas_squares[y][x]
                self.canvas.itemconfigure(square, fill = 'black', outline = 'black')
        self.image_numpy = np.zeros(shape = (64, 64), dtype = np.uint8)
        self.result.set(' ')

    def load_picture(self):
        self.result.set(' ')
        filename = filedialog.askopenfilename()
        if filename:
            image = Image.open(filename)
            if image.size[0] != 64 or image.size[1] != 64:
                image = image.resize((64, 64), Image.Resampling.LANCZOS)
            self.image_numpy = np.array(image.convert('L'))  # Convert to grayscale
            self.draw_picture()

    def draw_picture(self):
        for y in range(64):
            for x in range(64):
                square = self.canvas_squares[y][x]
                color = 'white' if self.image_numpy[y][x] > 128 else 'black'
                self.canvas.itemconfigure(square, fill=color, outline=color)

    def canvas_draw(self, evt):
        x = evt.x // 5
        y = evt.y // 5
        if 0 <= x < 64 and 0 <= y < 64:
            square = self.canvas_squares[y][x]
            color = 'black' if self.IS_ERASE else 'white'
            self.canvas.itemconfigure(square, fill = color, outline = color)
            self.image_numpy[y][x] = 0 if self.IS_ERASE else 255

    def canvas_mouse_down(self, evt):
        self.canvas_draw(evt)

    def canvas_mouse_move(self, evt):
        self.canvas_draw(evt)

    def recognize_char(self):
        # Preprocess the drawn or loaded image
        processed_img = self.image_numpy.copy()  

        # Convert to uint8 if it's not already (normalize to 0-255)
        if processed_img.dtype != np.uint8:
            processed_img = (processed_img * 255).astype(np.uint8)

        # Ensure the image is grayscale 
        if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        # Resize to 64x64
        processed_img = cv2.resize(processed_img, (64, 64))

        # Normalize for model (0 to 1)
        processed_img_norm = processed_img.astype('float32') / 255.0

        # Reshape for model input (add batch and channel dimensions)
        char_img_input = processed_img_norm.reshape(1, 64, 64, 1)

        # Create a Classifier instance 
        classifier = Classifier(self.model_path, None, None, character_dict, None, True)

        # Call the single character recognition method
        predicted_class, character, confidence = classifier.recognize_single_character(char_img_input)

        # Update the result
        self.result.set(f'Розпізнано: {character}\nТочність: {confidence:.3f}')

if __name__ == "__main__":
    main_window = MainWindow()
    main_window.mainloop()