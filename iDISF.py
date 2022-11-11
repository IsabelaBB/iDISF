#################################
# Created by IB Barcelos
# 2021
##################################

from idisf import iDISF_scribbles
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedStyle
import cv2
import os
import math
from scipy import ndimage
import higra as hg
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.pixel_data_handlers import convert_color_space
from cv2.ximgproc import createStructuredEdgeDetection


try:
    from utils import imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())


def safe_cast(val, to_type, default=None):
    try:
      return to_type(val)
    except (ValueError, TypeError):
      return default

def validValue(val, valMin, valMax):
  if(val < valMin):
    val = valMin
  elif(val > valMax):
    val = valMax
  return val


class DicomImage():
  def __init__(self, imagePath):
    self.data = pydicom.dcmread(imagePath)
    self.imagePath = imagePath

  def convertImage(self):
    arr = self.data.pixel_array.astype(np.float64) # get numpy array as representation of image data

    # pixel_array seems to be the original, non-rescaled array.
    # If present, window center and width refer to rescaled array
    # -> do rescaling if possible.
    if ('RescaleIntercept' in self.data) and ('RescaleSlope' in self.data):
      intercept = self.data.RescaleIntercept  # single value
      slope = self.data.RescaleSlope
      arr = slope * arr + intercept

    # get default window_center and window_width values
    wc = (arr.max() + arr.min()) / 2.0
    ww = arr.max() - arr.min() + 1.0

    # overwrite with specific values from data, if available
    if ('WindowCenter' in self.data) and ('WindowWidth' in self.data):
      ew = self.data['WindowWidth']
      ec = self.data['WindowCenter']
      wc = int(ew.value[0] if ew.VM > 1 else ew.value)
      ww = int(ec.value[0] if ec.VM > 1 else ec.value)
    else:
      wc = (arr.max() + arr.min()) / 2.0
      ww = arr.max() - arr.min() + 1.0

    return self.get_PGM_from_numpy_arr(arr, wc, ww)

  def get_PGM_from_numpy_arr(self, arr, wc, ww):
    """Given a 2D numpy array as input write
    gray-value image data in the PGM
    format into a byte string and return it.
    arr: single-byte unsigned int numpy array
    note: Tkinter's PhotoImage object seems to
    accept only single-byte data
    """

    if arr.dtype != np.uint8:
        raise ValueError
    if len(arr.shape) != 2:
        raise ValueError

    # array.shape is (#rows, #cols) tuple; PGM input needs this reversed
    col_row_string = ' '.join(reversed([str(x) for x in arr.shape]))

    bytedata_string = '\n'.join(('P5', col_row_string, str(arr.max()),
                                 arr.tostring()))
    return bytedata_string

  def histogram_equalization(self, img, no_bins):
    
    #img- the image as a numpy.array
    #the appropriate number of bins, `no_bins` in the histogram is chosen by experiments, 
    #until the contrast is convenient
    
    image_hist, bins = np.histogram(img.flatten(), no_bins, normed=True)
    csum = image_hist.cumsum() 
    cdf_mult = np.max(img) * csum / csum[-1] # cdf multiplied by a factor

    #  linear interpolation of cdf_mult to get new pixel values
    im_new = np.interp(img.flatten(), bins[:-1],  cdf_mult)

    return im_new.reshape(img.shape), cdf_mult
  
  def get_pl_image(self, hist_equal=False, no_bins=None):
    #dicom_filename- a string 'filename.dcm'
    #no_bins is the number of bins for histogram when hist_equal=False, else it is None
    #returns the np.array that defines the z-value for the heatmap representing the dicom image
    
    dic_file=pydicom.read_file(self.imagePath)
    img=dic_file.pixel_array#get the image as a numpy.array
    if hist_equal and isinstance(no_bins, int):
        img_new=self.histogram_equalization(img, no_bins)
        img_new = img_new[0]
        img_new=np.array(img_new, dtype=np.int16)
        return np.flipud(img_new)
    else:
        return np.flipud(img)
  
  def getPILImage2(self):
    image=self.get_pl_image(hist_equal=True, no_bins=255*16)

    print("MIN: ", np.min(image), " MAX: ", np.max(image))
    im = Image.fromarray(image)
    return im
  
  def getPILImage(self, brightness_factor = 1.0):
    """Get Image object from Python Imaging Library(PIL)
    Manipulate image brightness using brightness_factor parameter,
       receives a float value,
       Default = 1.0
       Brighter > 1.0 | Darker < 1.0 
    """
    if ('PixelData' not in self.data):
        raise TypeError("Cannot show image -- DICOM dataset does not have "
                        "pixel data")

    """
    bits = self.data.BitsAllocated
    samples = self.data.SamplesPerPixel
    
    if bits == 8 and samples == 1:
      mode = "L"
    elif bits == 8 and samples == 3:
      mode = "RGB"
    elif bits == 16:
      mode = "I;16"
    else:
      raise TypeError("Don't know PIL mode for %d BitsAllocated "
                        "and %d SamplesPerPixel" % (bits, samples))
    """

    #LUTification returns ndarrays
    #can only apply LUT if pydicom is installed
    image = self.get_LUT_value(self.data)

    # PIL size = (width, height)
    #size = (self.data.Columns, self.data.Rows)
    
    try:
        MIN = np.min(image)
        MAX = np.max(image)

        image = (image - MIN)*((255.0 - 0)/(MAX - MIN))
        #image = ((image - MIN)/np.max(image)) * 255
        image = image.astype(np.uint8)
        im = Image.fromarray(image)#.convert(mode)

    except:
        #When pixel data has multiple frames, output the first one
        MIN = np.min(image[0])
        MAX = np.max(image[0])
        
        image = (image[0] - MIN)*((255.0 - 0)/(MAX - MIN))
        #image = ((image - MIN)/np.max(image)) * 255
        image = image.astype(np.uint8)
        im = Image.fromarray(image)#.convert(mode)

    return im

  def get_LUT_value(self, data):
    return apply_modality_lut(data.pixel_array,data)
    

class AutoScrollbar(ttk.Scrollbar):
  """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
  def set(self, lo, hi):
    if float(lo) <= 0.0 and float(hi) >= 1.0:
      self.grid_remove()
    else:
      self.grid()
      ttk.Scrollbar.set(self, lo, hi)

  def pack(self, **kw):
    raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

  def place(self, **kw):
    raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

#########################################################################
#     Default Windows class
#########################################################################
class DefaultWindow:
  def __init__(self, root, titlePage, *args, **kwargs):
    # init app
    self.root = root # initialize root window
    
    try:
      self.style = ThemedStyle()
      self.style.set_theme('yaru')
      self.usingTtkthemes = True
    except Exception as inst:
      self.usingTtkthemes = False
    
    self.root.title(titlePage)  # set window title
    self.root.protocol('WM_DELETE_WINDOW', self.destructor)
    self.root.config(cursor="arrow")
    self.close = False
    self.fill = tk.N+tk.S+tk.E+tk.W

  def createWidgets(self, frame):
    top=frame.winfo_toplevel() 
    top.rowconfigure(0, weight=1) 
    top.columnconfigure(0, weight=1) 
    frame.rowconfigure(0, weight=1) 
    frame.columnconfigure(0, weight=1) 

  def destructor(self):
    #""" Destroy the root object and release all resources """
    self.root.destroy()
    self.close = True

  def hide(self):
    self.root.withdraw()

  def show(self):
    self.root.update()
    self.root.deiconify()



class DefaultImageWindow(DefaultWindow):
  def __init__(self, root, titlePage, settingsPage, *args, **kwargs):
    ##-----------------------------------------
    ## call init method of parent class
    DefaultWindow.__init__(self, root, titlePage, *args, **kwargs)
    ##-----------------------------------------

    self.appSettings = settingsPage
    self.titlepage = titlePage

    if(self.titlepage != 'Original Image'):
      self.root.protocol('WM_DELETE_WINDOW', self.hide)
    
    # variables
    self.currentImage = None # current canvas image in PIL format and no zoomed
    self.scale_img = 1.0 # image scale
    self.imgtk = None # current canvas image in ImageTk.PhotoImage format
    self.image_on_canvas = None # used to avoid the garbage collector in image canvas 
    self.imageName = None

    #### MAIN FRAME
    self.frame = ttk.Frame(master=self.root, relief=tk.RAISED, borderwidth=0)
    self.frame.grid(row=0, column=0, sticky=self.fill, padx=2, pady=2)
    self.createWidgets(self.frame)

    #### MENU
    self.__initMenuFrame__()

    #### CANVAS FRAME TO SHOW IMAGES
    self.__initCanvasFrame__(self.frame, 0, 0)

    #### BIND EVENTS
    self.__initBindEvents__()

  #### init frame blocks
  def __initMenuFrame__(self):
    """
    Create the menu commands
    """
    self.menu = tk.Menu(self.root)
    self.root.config(menu=self.menu)

    self.zoomMenu = tk.Menu(self.menu, tearoff=True)
    self.zoomMenu.add_command(label="Zoom in  100%     CTRL+ +", command=self.zoom_in_big)
    self.zoomMenu.add_command(label="Zoom in  10%            +", command=self.zoom_in_small)
    self.zoomMenu.add_command(label="Zoom out 100%     CTRL+ -", command=self.zoom_out_big)
    self.zoomMenu.add_command(label="Zoom out 10%            -", command=self.zoom_out_small)
    self.menu.add_cascade(label="Zoom", menu=self.zoomMenu)

    if(self.titlepage != 'Original image'):
      self.menu.add_command(label="Save Image", command=self.saveImage)
    
  def __initCanvasFrame__(self, master, row, column):
    """
    Create the canvas frames to show images
    """
    self.canvasLabel = ttk.Frame(master=master)
    self.canvasLabel.grid(row=row, column=column, columnspan=1, sticky=self.fill, padx=2, pady=2)
    self.createWidgets(self.canvasLabel)
    
    self.frame.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
    self.frame.columnconfigure(0, weight=1)
    
    self.canvas = tk.Canvas(master=self.canvasLabel)
    self.canvas.grid(row=0, column=0,sticky='')

    #### Vertical and horizontal scrollbars for canvas
    self.scrollbarImageY = AutoScrollbar(self.canvasLabel, orient='vertical')
    self.scrollbarImageX = AutoScrollbar(self.canvasLabel, orient='horizontal')
    
    self.scrollbarImageY.grid(row=0, column=1, sticky='ns')
    self.scrollbarImageX.grid(row=1, column=0, sticky='we')
    
    self.scrollbarImageY.config(command=self.canvas.yview)
    self.scrollbarImageX.config(command=self.canvas.xview)
    self.canvas.configure(yscrollcommand=self.scrollbarImageY.set, xscrollcommand=self.scrollbarImageX.set)
    self.createWidgets(self.canvas)

    # Assign the region to be scrolled 
    self.canvas.config(scrollregion=self.canvas.bbox('all'))
    
  
  def __initBindEvents__(self):
    """
    Create mouse click and key events
    """
    self.canvas.bind('<Button-5>', self.zoom_out_small)  # zoom for Linux, wheel scroll down
    self.canvas.bind('<Button-4>', self.zoom_in_small)  # zoom for Linux, wheel scroll up
    
    self.root.bind("<KP_Add>", self.zoom_in_small) # zoom with <+> key
    self.root.bind("<KP_Subtract>", self.zoom_out_small) # zoom with <-> key
    self.root.bind("<Control-KP_Add>", self.zoom_in_big) # zoom with <ctrl>+<+> keys
    self.root.bind("<Control-KP_Subtract>", self.zoom_out_big) # zoom with <ctrl>+<-> keys
    
    self.canvas.bind("<MouseWheel>", self.zoom_windowsSO)
    self.root.bind("<plus>", self.zoom_in_small) # zoom with <+> key
    self.root.bind("<minus>", self.zoom_out_small) # zoom with <-> key
    self.root.bind("<Control-plus>", self.zoom_in_big) # zoom with <ctrl>+<+> keys
    self.root.bind("<Control-minus>", self.zoom_out_big) # zoom with <ctrl>+<-> keys


  ### ZOOM FUNCTIONS
  def zoom_in_big(self, event=None):
    """
    Zoom in 50%
    """
    if(event is None):
      self.zoom(1.5)
    else:
      self.zoom(1.5, event.x, event.y)

  def zoom_in_small(self, event=None):
    """
    Zoom in 10%
    """
    if(event is None):
      self.zoom(1.1)
    else:
      self.zoom(1.1, event.x, event.y)

  def zoom_out_big(self, event=None):
    """
    Zoom out 50%
    """
    if(event is None):
      self.zoom(0.5)
    else:
      self.zoom(0.5, event.x, event.y)

  def zoom_out_small(self, event=None):
    """
    Zoom out 10%
    """
    if(event is None):
      self.zoom(0.9)
    else:
      self.zoom(0.9, event.x, event.y)
  
  def zoom_windowsSO(self, event):
    if event.num == 5 or event.delta == -120:
      self.zoom_out_small()
    if event.num == 4 or event.delta == 120:
      self.zoom_in_small()

  def zoom(self, value, x=None, y=None):
    """
    Main zoom function: calculates the new scale and updated the image
    INPUT: 
      value : zoom scale
      x,y : mouse coordinate 
    """
    min_size = 900
    max_size = 10000000
    width, height = self.currentImage.size
    size = self.scale_img * value * width * self.scale_img * value * height
    if(size < min_size or size > max_size):
      return

    self.scale_img *= value
    self.image2Tk()


  ### IMAGE FUNCTIONS
  def saveImage(self):
    """
    Saves the image shown on canvas
    """
    if(self.currentImage is not None):
      folder_selected = filedialog.asksaveasfilename(initialdir = "./",title = "Save file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"),("all files","*.*")))
      if(len(folder_selected) != 0):
        if(self.currentImage.mode != 'RGB'):
          image = self.currentImage.convert('RGB')
          image.save(folder_selected)
        else:
          self.currentImage.save(folder_selected)
        
    else:
      message = messagebox.showerror("Error", "No image founded.")
      return
  
  def image2Tk(self):
    """
    convert a PIL image to a tkinter image
    """
    width, height = self.currentImage.size
    self.canvas.delete('all')
    self.canvas_rect = None
    self.canvas_center_rect = None
    self.canvas_lines = []

    if(self.scale_img == 1):
      self.imgtk = ImageTk.PhotoImage(image=self.currentImage)
    else:
      self.imgtk = ImageTk.PhotoImage(self.currentImage.resize((int(width*self.scale_img), int(height*self.scale_img)), Image.ANTIALIAS))

    self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.imgtk)
    self.canvas.image = self.imgtk
    self.canvas.config(width=self.imgtk.width(), height=self.imgtk.height())
    self.canvas.config(scrollregion=self.canvas.bbox('all'))

  def setImage(self, imageName, img=None):
    """
    Change the current image shown on canvas and reset the image scale
    """
    self.imageName = imageName
    if(img is not None):
      self.currentImage = img
    elif(os.path.isfile(imageName) is True):
      try:
        self.currentImage = Image.open(self.imageName)
      except Exception as inst:
        self.currentImage = None
    else:
      self.destructor()

    if(self.currentImage is None):
      self.destructor()
    else:
      self.image2Tk()
      self.scale_img = 1.0

  '''  
  def readImage(self, imageName):
    self.imageName = imageName
    self.currentImage = Image.open(self.imageName)
    if(self.currentImage is None):
      self.destructor()
    else:
      self.image2Tk()
      self.scale_img = 1.0
  
  def setImage(self, img, imageName):
    self.imageName = imageName
    self.currentImage = img
    if(self.currentImage is None):
      self.destructor()
    else:
      self.image2Tk()
      self.scale_img = 1.0
  '''

class PopupSpinbox(DefaultWindow):
  def __init__(self, root, titlePage, question, defaultAnswer, maxVal, *args, **kwargs):
    ##-----------------------------------------
    ## call init method of child class
    DefaultWindow.__init__(self, root, titlePage, *args, **kwargs)
    ##-----------------------------------------

    self.answer = tk.StringVar()
    self.intAnswer = defaultAnswer
    self.setAnswer = False
    self.maxVal = maxVal

    # tk objects
    self.frame = ttk.Frame(master=self.root, relief=tk.RAISED, borderwidth=0)
    self.frame.grid(row=0, column=0, sticky=self.fill, padx=2, pady=2)
    self.createWidgets(self.frame)

    self.labelEntry = ttk.Label(self.frame, text=question+' (min 0, max '+str(maxVal)+')')
    self.labelEntry.grid(row=0, column=0, sticky=self.fill, padx=2, pady=2)

    self.answer.set(defaultAnswer)
    self.spinbox = tk.Spinbox(master=self.frame, from_=0, to=maxVal, textvariable=self.answer)
    self.spinbox.grid(row=1, column=0, sticky=self.fill, padx=2, pady=2)
    
    self.buttonOk = ttk.Button(master=self.frame, text='Ok', command=self.sendAnswer)
    self.buttonOk.grid(row=2, column=0, sticky=self.fill, padx=2, pady=2)

    self.buttonCancel = ttk.Button(master=self.frame, text='Cancel', command=self.destructor)
    self.buttonCancel.grid(row=2, column=1, sticky=self.fill, padx=2, pady=2)

  def validAnswer(self):
    if(self.answer.get().isdigit()):
      value = safe_cast(safe_cast(self.answer.get(), float, 0), int, 0) # convert : string -> float -> int
      if(value >= 0 and value <= self.maxVal):
        self.intAnswer = value
        return True
    return False

  def sendAnswer(self):
    if(self.validAnswer()):
      self.setAnswer = True
      self.destructor()
    else:
      message = messagebox.showerror("Error", "Invalid value.")


#########################################################################
#     Windows settings class
#########################################################################
class SettingsWindow(DefaultWindow):
  def __init__(self, root, *args, **kwargs):
    ##-----------------------------------------
    ## call init method of child class
    DefaultWindow.__init__(self, root, "Settings", *args, **kwargs)
    ##-----------------------------------------

    # init app
    self.root.minsize(285, 520)

    # variables
    self.bordersFileName = ''
    self.segmFileName = ''
    self.labelsFileName = ''


    if(os.path.isfile("./opencv_sed_model.yml.gz")):
      self.detector = createStructuredEdgeDetection("./opencv_sed_model.yml.gz")
    else:
      self.detector = createStructuredEdgeDetection(get_sed_model_file())
    
    # other windows
    self.appImage = None
    self.appSegm = None
    self.appBorders = None
    self.appLabels = None
    self.winSegm = None
    self.winBorders = None
    self.winLabels = None

    # tk objects
    self.icondelete = None
    self.BtntextDelete = None
    if(os.path.isfile("./python3/delete.png") is True):
      self.icondelete = tk.PhotoImage(file = "./python3/delete.png").subsample(22, 32) 
    else:
      self.Btntext='Delete all'

    self.iconUndo = None
    self.BtntextUndo = None
    if(os.path.isfile("./python3/undo.png") is True):
      self.iconUndo = tk.PhotoImage(file = "./python3/undo.png").subsample(40, 40) 
    else:
      self.BtntexUndo='Undo'
    
    self.BtntextSave = None
    self.iconSave = None
    if(os.path.isfile("./python3/save.png") is True):
      self.iconSave = tk.PhotoImage(file = "./python3/save.png").subsample(20, 20) 
    else:
      self.BtntextSave='Save'

    self.BtntextErase = None
    self.iconErase = None
    if(os.path.isfile("./python3/erase.png") is True):
      self.iconErase = tk.PhotoImage(file = "./python3/erase.png").subsample(10, 10) 
    else:
      self.BtntextErase='Erase'


    row = 0
    self.frame = ttk.Frame(master=self.root, borderwidth=1)
    self.frame.grid(row=row, column=0, columnspan=1, sticky=self.fill, padx=4, pady=4)
    self.createWidgets(self.frame)

    self.__initMenu_windSettings__()
    self.__initSettingFrame_windSettings__(self.frame, 0, 0, 1)
    self.__initShowFrame_windSettings__(self.frame, 1, 0, 1)
    self.__initBindEvents_windSettings__()


  #### init frame blocks
  def __initMenu_windSettings__(self):
    self.menu = tk.Menu(self.root)
    self.root.config(menu=self.menu)

    self.menu.add_command(label="Load Folder", command=self.loadDir)
    self.menu.add_command(label="Load Image", command=self.loadFile)


  def __initSettingFrame_windSettings__(self, master, row, column, columnspan):
    # variables
    self.maxScribbleSize = 50
    self.minScribbleSize = 4

    ######### define main frame and notebook tabs
    self.confLabel = ttk.LabelFrame(master=master, text='Segmentation')
    self.confLabel.grid(row=row, column=column, columnspan=columnspan, sticky=self.fill, padx=4, pady=2)
    parent = self.confLabel
    self.createWidgets(self.confLabel)
    
    self.tabControl = ttk.Notebook(parent, width=260) 
    self.tab1 = ttk.Frame(self.tabControl) 
    self.tab2 = ttk.Frame(self.tabControl) 

    self.tabControl.add(self.tab1, text ='iDISF') 
    self.tabControl.add(self.tab2, text ='Watershed') 
    self.tabControl.grid(row=0, column=0, columnspan=1, sticky=self.fill, padx=4, pady=2)
    row=0

    ####### TAB1 : iDISF
    parent = self.tab1
    self.__initiDISFFrame_windSettings__(parent, row, 0, 10)
    ##### TAB 2 : Watershed with Higra
    parent = self.tab2
    row=0
    self.__initWSFrame_windSettings__(parent, row, 0, 20)

    ### scribbles frame
    self.__initScribblesFrame_windSettings__(self.confLabel, 1, 0, 1)

    ##### segmentation Button
    self.runSegmBtn = ttk.Button(master=self.confLabel, text = 'Run Segmentation', command = self.runSegm)
    self.runSegmBtn.grid(row=2, column=0, columnspan=1, sticky=self.fill, padx=4, pady=4)


  def __initiDISFFrame_windSettings__(self, master, row, column, columnspan):
    # variables
    self.c1 = tk.StringVar()
    self.c2 = tk.StringVar()
    self.function = tk.StringVar()

    self.n0 = tk.StringVar()
    self.n0Min = 0
    self.n0Max = 8000
    self.n0Range = self.n0Max - self.n0Min
    
    self.iterations = tk.StringVar()
    self.itMin = 1
    self.itMax = 20
    self.itRange = self.itMax - self.itMin

    self.idisfMethodList = ["Seed removal by class", "Seed removal by relevance"] 
    self.functionList = ["Color distance", "Gradent variation", "Gradient variation norm.", "Sum gradient variation"] 
    self.bordersValue = tk.IntVar() 

    ## GRID Seeds 
    self.n0Label = ttk.Label(master=master)
    self.n0Label.grid(row=row, column=column, columnspan=columnspan, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.n0Label)
    row+=1

    self.n0LabelName = ttk.Label(master=self.n0Label, text='GRID Seeds', width=10)
    self.n0LabelName.grid(row=0, column=0, sticky=self.fill, padx=0, pady=0)
    
    if(self.usingTtkthemes):
      self.n0Scale = ttk.Scale(master=self.n0Label, from_ = self.n0Min, to = self.n0Max, orient = tk.HORIZONTAL, command = self.setN0, length=115)
    else:
      self.n0Scale = tk.Scale(master=self.n0Label, from_ = self.n0Min, to = self.n0Max, orient = tk.HORIZONTAL, command = self.setN0, resolution = 1, digits = 1, showvalue = 0, sliderlength=20, length=115)
  
    self.n0Scale.grid(row=0, column=1, columnspan=8, sticky=self.fill, padx=2, pady=0)

    self.n0Entry = ttk.Entry(master=self.n0Label, textvariable=self.n0, width=5)
    self.n0Entry.grid(row=0, column=9, sticky=self.fill, padx=0, pady=0)
    self.setN0(str(self.n0Scale.get()))

    
    ### iterations
    self.itLabel = ttk.Label(master=master)
    self.itLabel.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.itLabel)
    row+=1

    self.itLabelName = ttk.Label(master=self.itLabel, text='Iterations')
    self.itLabelName.grid(row=0, column=0, columnspan=1, sticky=self.fill, padx=0, pady=0)
    
    if(self.usingTtkthemes):
      self.itScale = ttk.Scale(master=self.itLabel, from_ = self.itMin, to = self.itMax, orient = tk.HORIZONTAL, command = self.setIt, length=115)
    else:
      self.itScale = tk.Scale(master=self.itLabel, from_ = self.itMin, to = self.itMax, orient = tk.HORIZONTAL, command = self.setIt, resolution = 1, digits = 1, showvalue = 0, sliderlength=20, length=115)
    self.itScale.grid(row=0, column=1, columnspan=18, sticky=self.fill, padx=2, pady=0)
 
    self.itEntry = ttk.Entry(master=self.itLabel, textvariable=self.iterations, width=5)
    self.itEntry.grid(row=0, column=19, sticky=self.fill, padx=0, pady=0)
    self.setIt(str(self.itScale.get()))

    ### c1
    self.c1Label = ttk.Label(master=master)
    self.c1Label.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.c1Label)
    row+=1

    self.c1LabelName = ttk.Label(master=self.c1Label, text='c1')
    self.c1LabelName.grid(row=0, column=0, columnspan=1, sticky=self.fill, padx=0, pady=0)
    
    if(self.usingTtkthemes):
      self.c1Scale = ttk.Scale(master=self.c1Label, from_ = 10, to = 100, orient = tk.HORIZONTAL, command = self.setC1, length=115)
    else:
      self.c1Scale = tk.Scale(master=self.c1Label, from_ = 10, to = 100, orient = tk.HORIZONTAL, command = self.setC1, resolution = 1, digits = 1, showvalue = 0, sliderlength=20, length=115)

    self.c1Scale.grid(row=0, column=1, columnspan=18, sticky=self.fill, padx=2, pady=0)
    self.c1Entry = ttk.Entry(master=self.c1Label, textvariable=self.c1, width=5)
    self.c1Entry.grid(row=0, column=19, sticky=self.fill, padx=0, pady=0)
    self.setC1("7")

    ### c2
    self.c2Label = ttk.Label(master=master)
    self.c2Label.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.c2Label)
    row+=1

    self.c2LabelName = ttk.Label(master=self.c2Label, text='c2')
    self.c2LabelName.grid(row=0, column=0, columnspan=1, sticky=self.fill, padx=0, pady=0)

    if(self.usingTtkthemes):
      self.c2Scale = ttk.Scale(master=self.c2Label, from_ = 10, to = 100, orient = tk.HORIZONTAL, command = self.setC2, length=115)
    else:
      self.c2Scale = tk.Scale(master=self.c2Label, from_ = 10, to = 100, orient = tk.HORIZONTAL, command = self.setC2, resolution = 1, digits = 1, showvalue = 0, sliderlength=20, length=115)
    
    self.c2Scale.grid(row=0, column=1, columnspan=18, sticky=self.fill, padx=2, pady=0)
    self.c2Entry = ttk.Entry(master=self.c2Label, textvariable=self.c2, validate='focusout', width=5)
    self.c2Entry.grid(row=0, column=19, sticky=self.fill, padx=0, pady=0)
    self.setC2("8")

    # Functions
    self.funcLabel = ttk.Label(master=master)
    self.funcLabel.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.funcLabel)
    row+=1

    self.funcLabelName = ttk.Label(master=self.funcLabel, text='Function')
    self.funcLabelName.grid(row=0, column=0, columnspan=10, sticky=self.fill, padx=0, pady=2)
    self.funcList = ttk.Combobox(master=self.funcLabel, values=self.functionList, state="readonly")
    self.funcList.grid(row=0, column=10, columnspan=10, sticky=self.fill, padx=0, pady=2)
    self.funcList.current(0)
    self.setFunction(1)
    
    # Segm. Methods
    self.optLabel = ttk.Label(master=master)
    self.optLabel.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.optLabel)
    row+=1

    self.optLabelName = ttk.Label(master=self.optLabel, text='Method')
    self.optLabelName.grid(row=0, column=0, columnspan=10, sticky=self.fill, padx=0, pady=0)
    self.optList = ttk.Combobox(master=self.optLabel, values=self.idisfMethodList, state="readonly")
    self.optList.grid(row=0, column=10, columnspan=10, sticky=self.fill, padx=0, pady=0)
    self.optList.current(0)

    # Borders
    self.bordersLabelName = ttk.LabelFrame(master=master, text='Borders between')
    self.bordersLabelName.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=6, pady=4)
    self.createWidgets(self.bordersLabelName)
    row+=1

    if(self.usingTtkthemes):
      self.bordersRadio1 = ttk.Radiobutton(master=self.bordersLabelName, text='Regions', variable=self.bordersValue, value=0) 
      self.bordersRadio2 = ttk.Radiobutton(master=self.bordersLabelName, text='Trees', variable=self.bordersValue, value=1)
    else:
      self.bordersRadio1 = tk.Radiobutton(master=self.bordersLabelName, text='Regions', variable=self.bordersValue, value=0) 
      self.bordersRadio2 = tk.Radiobutton(master=self.bordersLabelName, text='Trees', variable=self.bordersValue, value=1)

    self.bordersRadio1.grid(row=0, column=0, columnspan=1, sticky=tk.W, padx=4, pady=1)
    self.bordersRadio2.grid(row=0, column=1, columnspan=1, sticky=tk.W, padx=4, pady=1)
    self.bordersValue.set(0)


  def __initWSFrame_windSettings__(self, master, row, column, columnspan):
    # variables
    self.weightFunctionList = ["L0", "L1", "L2", "L2_squared", "L_infinity", "max", "mean", "min"]
    self.wsAttributeList = ['Area', 'Volume', 'Dynamics']

    # Weight Function
    self.weightFunctionLabel = ttk.Label(master=master)
    self.weightFunctionLabel.grid(row=row, column=column, columnspan=columnspan, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.weightFunctionLabel)
    row+=1

    self.weightFunctionLabelName = ttk.Label(master=self.weightFunctionLabel, text='Weight Function')
    self.weightFunctionLabelName.grid(row=0, column=0, columnspan=10, sticky=self.fill, padx=2, pady=2)
    self.weightFunctionCombobox = ttk.Combobox(master=self.weightFunctionLabel, values=self.weightFunctionList, state="readonly", width=15)
    self.weightFunctionCombobox.grid(row=0, column=10, columnspan=10, sticky=self.fill, padx=2, pady=2)
    self.weightFunctionCombobox.current(0)

    # Watershed Attribute
    self.wsAttributeLabel = ttk.Label(master=master)
    self.wsAttributeLabel.grid(row=row, column=0, columnspan=20, sticky=self.fill, padx=4, pady=2)
    self.createWidgets(self.wsAttributeLabel)
    row+=1

    self.wsAttributeLabelName = ttk.Label(master=self.wsAttributeLabel, text='Attribute')
    self.wsAttributeLabelName.grid(row=0, column=0, columnspan=10, sticky=self.fill, padx=2, pady=2)
    self.wsAttributeCombobox = ttk.Combobox(master=self.wsAttributeLabel, values=self.wsAttributeList, state="readonly", width=15)
    self.wsAttributeCombobox.grid(row=0, column=10, columnspan=10, sticky=self.fill, padx=2, pady=2)
    self.wsAttributeCombobox.current(0)

    
  def __initScribblesFrame_windSettings__(self, master, row, column, columnspan):  
    # variables
    self.scribblesType = tk.IntVar() # 0 -> obj ; 1 -> fg
      
    ######### Scribbles
    self.scribblesTypeName = ttk.LabelFrame(master=master, text='Scribbles')
    self.scribblesTypeName.grid(row=row, column=column, columnspan=columnspan, sticky=self.fill, padx=6, pady=4)
    self.createWidgets(self.scribblesTypeName)

    self.scribblesRadioLabel = ttk.Label(master=self.scribblesTypeName)
    self.scribblesRadioLabel.grid(row=0, column=0, columnspan=20, sticky=self.fill, padx=0, pady=0)
    self.createWidgets(self.scribblesRadioLabel)

    auxLabel = ttk.Label(master=self.scribblesRadioLabel)
    auxLabel.grid(row=0, column=0, columnspan=10, sticky='w', padx=4, pady=1)
    
    auxLabel2 = ttk.Label(master=self.scribblesRadioLabel)
    auxLabel2.grid(row=1, column=0, columnspan=10, sticky='w', padx=4, pady=1)

    if(self.usingTtkthemes):
      self.scribblesObj = ttk.Radiobutton(master=auxLabel, text='Object', variable=self.scribblesType, value=0) 
      self.scribblesFg = ttk.Radiobutton(master=auxLabel2, text='Background', variable=self.scribblesType, value=1)
    else:
      self.scribblesObj = tk.Radiobutton(master=auxLabel, text='Object', variable=self.scribblesType, value=0) 
      self.scribblesFg = tk.Radiobutton(master=auxLabel2, text='Background', variable=self.scribblesType, value=1)

    self.scribblesObj.grid(row=0, column=0, columnspan=10, sticky=tk.W, padx=0, pady=0)
    self.delete1Btn = tk.Button(master=auxLabel, text=self.BtntextDelete, image = self.icondelete, command = self.clearObjMarkers)
    self.delete1Btn.grid(row=0, column=10, columnspan=10, sticky=tk.W, padx=0, pady=0)

    self.scribblesFg.grid(row=0, column=0, columnspan=10, sticky=tk.W, padx=0, pady=0)
    self.scribblesType.set(0)
    self.delete2Btn = tk.Button(master=auxLabel2, text=self.BtntextDelete, image = self.icondelete, command = self.clearBgMarkers)
    self.delete2Btn.grid(row=0, column=10, columnspan=10, sticky=tk.W, padx=0, pady=0)
    
    self.erase1Btn = tk.Radiobutton(master=self.scribblesRadioLabel, text='Erase', variable=self.scribblesType, value=3)
    self.erase1Btn.grid(row=2, column=0, columnspan=10, sticky='w', padx=2, pady=2)

    self.scribblesSizeLabelName = ttk.Label(master=self.scribblesRadioLabel, text='')
    self.scribblesSizeLabelName.grid(row=0, column=10, columnspan=10, rowspan=3, sticky='', padx=0, pady=0)
    
    if(self.usingTtkthemes):
      self.scribblesSizeScale = ttk.Scale(master=self.scribblesSizeLabelName, from_ = self.minScribbleSize, to = self.maxScribbleSize, orient = tk.VERTICAL, command = self.setScribblesSize, length=50)
    else:
      self.scribblesSizeScale = tk.Scale(master=self.scribblesSizeLabelName, from_ = self.minScribbleSize, to = self.maxScribbleSize, orient = tk.VERTICAL, command = self.setScribblesSize, resolution = 1, digits = 2, showvalue = 4, sliderlength=50, length=50)
    
    self.miniCanvas = tk.Canvas(master=self.scribblesSizeLabelName)
    #self.miniCanvas.config(width=16, height=16, bg="red")
    self.miniCanvas.config(width=20, height=20)
    self.miniCanvas_circle = None

    self.scribblesSizeScale.set(4)
    self.scribblesSizeScale.grid(row=1, column=0, sticky='', padx=0, pady=0)
    self.createWidgets(self.scribblesSizeScale)
    
    self.createWidgets(self.miniCanvas)
    self.miniCanvas.grid(row=2, column=0, sticky='', padx=0, pady=0)
    
    self.setScribblesSize(self.minScribbleSize)
      

  def __initShowFrame_windSettings__(self, master, row, column, columnspan):  
    # variables
    self.segm = tk.IntVar()
    self.segm.set(1)
    self.borders = tk.IntVar()
    self.borders.set(1)
    self.labels = tk.IntVar()
    self.labels.set(1)
      
    # save/show objects
    self.showLabelFrame = ttk.LabelFrame(master=master, text='Show/Save')
    self.showLabelFrame.grid(row=row, column=column, columnspan=columnspan, sticky=self.fill, padx=6, pady=4)
    self.createWidgets(self.showLabelFrame)

    auxLabel = ttk.Label(master=self.showLabelFrame)
    auxLabel.grid(row=0, column=0, columnspan=1, sticky='w', padx=4, pady=2)
    self.createWidgets(auxLabel)

    self.save1Btn = tk.Button(master=auxLabel, text=self.BtntextSave, image = self.iconSave, command = self.saveSegm)
    self.save1Btn.grid(row=0, column=0, columnspan=10, sticky=tk.W, padx=0, pady=1)
    
    if(self.usingTtkthemes):
      self.checkSegm = ttk.Checkbutton(master=auxLabel, text='Segmentation', command=self.setCheckSegm, variable=self.segm)
      self.checkBorders = ttk.Checkbutton(master=auxLabel, text='Borders', command=self.setCheckBorders, variable=self.borders)
      self.checkLabels = ttk.Checkbutton(master=auxLabel, text='Labels', command=self.setCheckLabels, variable=self.labels)
    else:
      self.checkSegm = tk.Checkbutton(master=auxLabel, text='Segmentation', command=self.setCheckSegm, variable=self.segm)
      self.checkBorders = tk.Checkbutton(master=auxLabel, text='Borders', command=self.setCheckBorders, variable=self.borders)
      self.checkLabels = tk.Checkbutton(master=auxLabel, text='Labels', command=self.setCheckLabels, variable=self.labels)
    
    self.checkSegm.grid(row=0, column=10, columnspan=10, sticky=tk.W, padx=0, pady=1)
    self.checkBorders.grid(row=1, column=10, columnspan=10, sticky=tk.W, padx=0, pady=1)
    self.checkLabels.grid(row=2, column=10, columnspan=10, sticky='w', padx=0, pady=1)

    self.save2Btn = tk.Button(master=auxLabel, text=self.BtntextSave, image = self.iconSave, command = self.saveBorders)
    self.save2Btn.grid(row=1, column=0, columnspan=10, sticky=tk.W, padx=0, pady=1)

    self.save3Btn = tk.Button(master=auxLabel, text=self.BtntextSave, image = self.iconSave, command = self.saveLabels)
    self.save3Btn.grid(row=2, column=0, columnspan=10, sticky=tk.W, padx=0, pady=1)


  def __initBindEvents_windSettings__(self):
    self.root.bind("<KeyRelease-c>", self.clearAndUpdateImage) # c key released: clear all scribbles
    self.tabControl.bind("<<NotebookTabChanged>>", self.on_tab_selected) # change tab
    self.n0Entry.bind('<Key-Return>', self.setTextN0) # buttom event: ENTER key in n0 text field
    self.itEntry.bind('<Key-Return>', self.setTextIt) # buttom event: ENTER key in iterations text field
    self.c1Entry.bind('<Key-Return>', self.setTextC1) # buttom event: ENTER key in c1 text field
    self.c2Entry.bind('<Key-Return>', self.setTextC2) # buttom event: ENTER key in c2 text field
    self.funcList.bind("<<ComboboxSelected>>", self.functionChange) # combo box event: function selected
    self.optList.bind("<<ComboboxSelected>>", self.methodChange) # combo box event: method selected
    self.weightFunctionCombobox.bind("<<ComboboxSelected>>", self.weightFunctionChangeEvent) # combo box event: weight selected
    self.wsAttributeCombobox.bind("<<ComboboxSelected>>", self.wsAttributeChangeEvent) # combo box event: attribute selected
      

  
  def clearMarkers(self):
    if(self.appImage is not None):
      self.appImage.clearMarkers()
      self.appImage.clearImage()
      self.appImage.image2Tk()  

  # buttom event: delete the object markers 
  def clearObjMarkers(self):
    if(self.appImage is not None):
      self.appImage.clearObjMarkers() 

  # buttom event: delete the object markers
  def clearBgMarkers(self):
    if(self.appImage is not None):
      self.appImage.clearBgMarkers()
 
  # c key released event: clear the markers and make a copy from the original image to the current image
  def clearAndUpdateImage(self, event):
    if(self.appImage is not None):
      self.appImage.clearAndUpdateImage(event)

  # combobox event: change the segmentation method
  def methodChange(self, event):
    if(self.optList.get() == 'Rem. by relevance'):
      self.setItMax(8000)
      self.setItMin(0)
      self.itLabelName['text'] = 'Final trees'
    else:
      self.setItMin(1)
      self.setItMax(20)
      self.itLabelName['text'] = 'Iterations'
      if(self.itScale.get() == 0):
        self.itScale.configure(from_=self.itMin)
        self.setIt("1")
        self.itScale.set(1)

  # combobox event: change the path-cost function method
  # ["Color distance", "Gradent variation", "Gradient variation norm.", "Sum gradient variation"] 
  def functionChange(self, event):
    if(self.funcList.get() == self.functionList[0]):
      self.setFunction(1)
    elif(self.funcList.get() == self.functionList[1]):
      self.setFunction(2)
    elif(self.funcList.get() == self.functionList[2]):
      self.setFunction(3)
    else:
      self.setFunction(4)
      

  def runSegm(self):
    if(self.tab_text == 'iDISF'):
      self.runiDISF()
    else:
      self.runWatershed()


  # call iDISF_scribbles function
  def runiDISF(self):
    if(self.appImage is None or self.appImage.close == True):
      message = messagebox.showerror("Error", "It's necessary to upload at least one image.")
      return

    markers = self.appImage.getMarkers()
    imgCV2 = self.appImage.getCurrentImage()
    marker_sizes = self.appImage.getMarkerSizes()
    
    if(len(markers) == 0):
      message = messagebox.showerror("Error", "It's necessary draw at least one scribble.")
      return

    if(self.appImage.objMarkers == 0):
      message = messagebox.showerror("Error", "It's necessary draw at least one object scribble.")
      return

    if(len(marker_sizes) == 1 and int(self.n0.get()) == 0):
      message = messagebox.showerror("Error", "Set more GRID seeds or draw at least two scribbles.")
      return

    segm_method = 1
    if(self.optList.get() == self.idisfMethodList[1]):
      segm_method = 2
      # no algoritmo já adiciona a nf o numero de pixels do marcador de objeto
      max_nf = int(self.n0.get()) 
      if(int(self.iterations.get()) > max_nf):
        message = messagebox.showerror("Error", "The maximum number of final GRID trees is %d."%(max_nf))
        return

    markers = np.array(markers)
    marker_sizes = np.array(marker_sizes)
    
    img = np.array(imgCV2)
    if(len(img.shape) == 2):
      img = np.stack((img,)*3, axis=-1)

    label_img = None
    border_img = None

    label_img, border_img = iDISF_scribbles(img, int(self.n0.get()), int(self.iterations.get()), markers, marker_sizes, self.appImage.objMarkers, int(self.function.get()), float(self.c1.get()), float(self.c2.get()), segm_method, self.bordersValue.get())
    self.destroyDefaultWindows()
    # label_img : 1 no objeto e 2 no fundo

    if(label_img is not None and border_img is not None and np.max(label_img) != 0):

      # label_img_bg : mascara com 0 no objeto e 255 no fundo
      if(np.max(label_img)-1 != 0):
        label_img_bg = (label_img-1) * (255/(np.max(label_img)-1)) 
      else:
        label_img_bg = label_img
      label_img_bg = np.uint8(label_img_bg)
      
      # label_img_bg : mascara com 255 no objeto e 0 no fundo
      label_img_fg = 255 - label_img_bg
      label_img_fg = np.uint8(label_img_fg)
      
      # label_img : 127 no objeto e 255 no fundo
      label_img = (label_img * 255)/np.max(label_img)
      label_img = np.uint8(label_img)

      # mascara para o foregorund
      img_fg = np.zeros([label_img.shape[0],label_img.shape[1],3],dtype=np.uint8)
      
      # aumenta o vermelho e o verde, mantem o azul
      img_fg[:,:,0] = img[:,:,0]*0.2
      img_fg[:,:,1] = img[:,:,1]*0.2

      img_fg[:,:,0] = cv2.bitwise_and(img_fg[:,:,0], label_img_fg)
      img_fg[:,:,1] = cv2.bitwise_and(img_fg[:,:,1], label_img_fg)
      img_fg[:,:,2] = cv2.bitwise_and(img[:,:,2], label_img_fg)

      # mascara para o background
      img_bg = np.zeros([label_img.shape[0],label_img.shape[1],3],dtype=np.uint8)

      # aumenta R e diminui G e B no background
      img_bg[:,:,1] = img[:,:,1]*0.2
      img_bg[:,:,2] = img[:,:,2]*0.2

      img_bg[:,:,0] = cv2.bitwise_and(img[:,:,0], label_img_bg)
      img_bg[:,:,1] = cv2.bitwise_and(img_bg[:,:,1], label_img_bg)
      img_bg[:,:,2] = cv2.bitwise_and(img_bg[:,:,2], label_img_bg)

      segm_img = cv2.bitwise_or(img_fg, img_bg)

      orig_bg = cv2.bitwise_and(img, img, mask=label_img_bg)
      only_fg_img = cv2.bitwise_or(img_fg, orig_bg)

      orig_fg = cv2.bitwise_and(img, img, mask=label_img_fg)
      only_bg_img = cv2.bitwise_or(img_bg, orig_fg)
      
      # convert from opencv to PIL
      border_img = Image.fromarray(border_img) 
      label_img = Image.fromarray(label_img) 
      segm_img = Image.fromarray(segm_img) 
      only_fg_img = Image.fromarray(only_fg_img) 
      only_bg_img = Image.fromarray(only_bg_img) 
      img = Image.fromarray(img) 

      self.appLabels,self.winLabels = self.newDefaultImageWindow(label_img, 'Labels', 'Labels', "Could not read the labeled image.")
      if(self.labels.get() == 0):
        self.appLabels.hide()

      self.appBorders,self.winBorders = self.newDefaultImageWindow(border_img, 'Borders', 'Borders', "Could not read the image borders.")
      if(self.borders.get() == 0):
        self.appBorders.hide()

      self.appSegm,self.winSegm = self.newSegmentationWindow(segm_img, only_fg_img, only_bg_img, img, 'Segmentation', 'Segmentation', "Could not read the image segmentation.")
      if(self.segm.get() == 0):
        self.appSegm.hide()
        
    else:
      message = messagebox.showerror("Error", "Could not read the image.")

    
  def runWatershed(self):
    if(self.appImage == None or self.appImage.close == True):
      message = messagebox.showerror("Error", "It's necessary to upload at least one image.")
      return

    original_image = np.array(self.appImage.getCurrentImage()) # converting to opencv
    if(len(original_image.shape) == 2):
      original_image = np.stack((original_image,)*3, axis=-1)
    

    # markers will store the user provided information: 
    # first channel (red) corresponds to background
    # second channel (green) corresponds to foreground
    markers = np.zeros_like(original_image)
    self.appImage.getMarkers()

    obj_coords = 0
    for size in self.appImage.markers_sizes[0:self.appImage.objMarkers]:
      obj_coords += size

    for [x,y] in self.appImage.markers[0:obj_coords]:
      markers[y-1:y+1, x-1:x+1, :]= (0,1,0) # green marker : object / white

    for [x,y] in self.appImage.markers[obj_coords::]:
      markers[y-1:y+1, x-1:x+1, :]= (1,0,0) # red marker : background / black

    # compute binary segmentation from the two markers
    label_img_fg = hg.binary_labelisation_from_markers(self.tree, markers[:,:,1], markers[:,:,0])
    label_img_fg = label_img_fg * 255
    label_img_fg = label_img_fg.astype(np.uint8)

    borders = cv2.bitwise_or(original_image, original_image, mask=label_img_fg)
    
    label_img_bg = cv2.bitwise_not(label_img_fg) # object = black , background = white
    
    sm = cv2.bitwise_or(self.sm, self.sm, mask=label_img_bg)
    borders = cv2.bitwise_or(borders, sm)

    label_img = 1 + label_img_bg/255
    label_img = (label_img * 255)/np.max(label_img)
    label_img = np.uint8(label_img)
    
    # mantem os canais da imagem, mas apenas no foreground
    img_fg = np.zeros([original_image.shape[0],original_image.shape[1],3],dtype=np.uint8)
    
    img = np.uint8(original_image)

    # aumenta o vermelho e o verde, mantem o azul
    img_fg[:,:,0] = img[:,:,0]*0.2
    img_fg[:,:,1] = img[:,:,1]*0.2

    img_fg[:,:,0] = cv2.bitwise_and(img_fg[:,:,0], label_img_fg)
    img_fg[:,:,1] = cv2.bitwise_and(img_fg[:,:,1], label_img_fg)
    img_fg[:,:,2] = cv2.bitwise_and(img[:,:,2], label_img_fg)

    # mascara para o background
    img_bg = np.zeros([label_img.shape[0],label_img.shape[1],3],dtype=np.uint8)

    # aumenta R e diminui G e B no background
    img_bg[:,:,1] = img[:,:,1]*0.2
    img_bg[:,:,2] = img[:,:,2]*0.2

    img_bg[:,:,0] = cv2.bitwise_and(img[:,:,0], label_img_bg)
    img_bg[:,:,1] = cv2.bitwise_and(img_bg[:,:,1], label_img_bg)
    img_bg[:,:,2] = cv2.bitwise_and(img_bg[:,:,2], label_img_bg)

    segm_img = cv2.bitwise_or(img_fg, img_bg)

    orig_bg = cv2.bitwise_and(img, img, mask=label_img_bg)
    only_fg_img = cv2.bitwise_or(img_fg, orig_bg)

    orig_fg = cv2.bitwise_and(img, img, mask=label_img_fg)
    only_bg_img = cv2.bitwise_or(img_bg, orig_fg)

    self.destroyDefaultWindows()

    # convert from opencv to PIL
    borders = Image.fromarray(borders) 
    label_img = Image.fromarray(label_img) 
    segm_img = Image.fromarray(segm_img) 
    only_fg_img = Image.fromarray(only_fg_img) 
    only_bg_img = Image.fromarray(only_bg_img) 
    img = Image.fromarray(img) 

    self.appLabels,self.winLabels = self.newDefaultImageWindow(label_img, 'Labels', 'Labels', "Could not read the labeled image.")
    if(self.labels.get() == 0):
      self.appLabels.hide()

    self.appBorders,self.winBorders = self.newDefaultImageWindow(borders, 'Borders', 'Borders', "Could not read the image borders.")
    if(self.borders.get() == 0):
      self.appBorders.hide()

    self.appSegm,self.winSegm = self.newSegmentationWindow(segm_img, only_fg_img, only_bg_img, img, 'Segmentation', 'Segmentation', "Could not read the image segmentation.")
    if(self.segm.get() == 0):
      self.appSegm.hide()


  # button event: Save image from appSegmentation window 
  def saveSegm(self):
    if(self.appSegm is not None):
      folder_selected = filedialog.asksaveasfilename(initialdir = "./",title = "Save file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"),("all files","*.*")))
      if(len(folder_selected) != 0):
        self.appSegm.currentImage.save(folder_selected)
    else:
      message = messagebox.showerror("Error", "No segmentation founded.")
      return

  # button event: Save image from appBorders window 
  def saveBorders(self):
    if(self.appBorders is not None):
      folder_selected = filedialog.asksaveasfilename(initialdir = "./",title = "Save file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"),("all files","*.*")))
      if(len(folder_selected) != 0):
        image = self.appBorders.currentImage.convert('RGB')
        image.save(folder_selected)
    else:
      message = messagebox.showerror("Error", "No image borders founded.")
      return

  # button event: Save image from appLabels window 
  def saveLabels(self):
    if(self.appLabels is not None):
      folder_selected = filedialog.asksaveasfilename(initialdir = "./",title = "Save file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"),("all files","*.*")))
      if(len(folder_selected) != 0):
        self.appLabels.currentImage.save(folder_selected)
    else:
      message = messagebox.showerror("Error", "No image labels founded.")
      return

  # checkbox event: Open or close the segmentation window
  def setCheckSegm(self):
    if(self.appSegm != None):
      if(self.appSegm.close):
        self.appSegm = None
    
      elif(self.segm.get() == 1):
        self.appSegm.show()

      else:
        self.appSegm.hide()

  # checkbox event: Open or close the borders window
  def setCheckBorders(self):
    if(self.appBorders != None):
      if(self.appBorders.close):
        self.appBorders = None

      elif(self.borders.get() == 1):
        self.appBorders.show()

      else:
        self.appBorders.hide()

  # checkbox event: Open or close the labels window
  def setCheckLabels(self):
    if(self.appLabels != None):
      if(self.appLabels.close):
        self.appLabels = None
    
      elif(self.labels.get() == 1):
        self.appLabels.show()

      else:
        self.appLabels.hide()

  
  # create and open a new window with the class DefaultPage
  def newDefaultImageWindow(self, img, imageName, titlePage, errorMessage):
    wind = tk.Toplevel(self.root)
    app = DefaultImageWindow(wind, titlePage, self)
    app.setImage(imageName, img)
    if(app is None or app.currentImage is None):
      message = messagebox.showerror("Error", errorMessage)
    return app,wind
  
  def newSegmentationWindow(self, segm_img, img_fg, img_bg, orig_img, imageName, titlePage, errorMessage):
    wind = tk.Toplevel(self.root)
    app = SegmentationWindow(wind, titlePage, self)
    app.setImage(segm_img, img_fg, img_bg, orig_img, imageName)
    if(app is None or app.currentImage is None):
      message = messagebox.showerror("Error", errorMessage)
    return app,wind

  # create and open a new window with the class AppImage
  def newAppImageWindow(self, files, errorMessage):
    wind = tk.Toplevel(self.root)
    app = AppImage(wind, self)
    self.appImage = app
    app.setImages(files)
    if(app is None):
      message = messagebox.showerror("Error", errorMessage)
    return app,wind


  # button event: Try load all imagens located in a folder selected by user
  def loadDir(self):
    folder_selected = filedialog.askdirectory()
    if(len(folder_selected) != 0):
      if(self.appImage is not None):
        self.appImage.destructor()
      self.destroyDefaultWindows()
      files = []
      [files.append(folder_selected+"/"+fileName) for fileName in os.listdir(folder_selected)]
      self.appImage,self.windImage = self.newAppImageWindow(files, "Could not read any image.")

  # button event: Try load all imagens selected by user
  def loadFile(self):
    folder_selected = filedialog.askopenfilenames()
    if(len(folder_selected) != 0):
      if(self.appImage is not None):
        self.appImage.destructor()
      self.destroyDefaultWindows()
      self.appImage,self.windImage = self.newAppImageWindow(folder_selected, "Could not read any image.")


  # destroy default (segmentation, borders and labels) windows
  def destroyDefaultWindows(self):
    if(self.appSegm != None):
      self.appSegm.destructor()
      self.appSegm = None
    if(self.appBorders != None):
      self.appBorders.destructor()
      self.appBorders = None
    if(self.appLabels != None):
      self.appLabels.destructor()
      self.appLabels = None
    

  # GRID seeds events
  # Entry event (text field)
  def setTextN0(self, event):
    value = safe_cast(safe_cast(self.n0.get(), float, 0), int, 0) # convert : string -> float -> int
    value = validValue(value, self.n0Min, self.n0Max)             # get a valid value
    self.n0.set(str(value)) # change text
    self.n0Scale.set(value) # change scale

  # Scale event: Change the current value of the GRID seeds
  def setN0(self, value):
    value = safe_cast(safe_cast(value, float, 0), int, 0) # convert : string -> float -> int
    value = validValue(value, self.n0Min, self.n0Max)             # get a valid value
    self.n0.set(str(value)) # change text


  def setScribblesSize(self, value):
    value = safe_cast(safe_cast(value, float, 0), int, 0) # convert : string -> float -> int
    value = validValue(value, self.minScribbleSize, self.maxScribbleSize)     # get a valid value
    
    if(self.miniCanvas_circle is not None):
      self.miniCanvas.delete(self.miniCanvas_circle)
    
    bbox = [8,8,20,20]
    center = (20/2, 20/2)
    
    # get perc. scale
    perc_scale = (value-self.minScribbleSize) / (self.maxScribbleSize-self.minScribbleSize)

    # circle ratio
    ratio = ((bbox[2]-bbox[0]) * perc_scale + 4) / 2

    self.miniCanvas_circle = self.miniCanvas.create_oval(center[0]-ratio, center[1]-ratio, center[0]+ratio, center[1]+ratio, fill="black", width=0)


  # Iterations events
  # Entry event (text field)
  def setTextIt(self, event):
    value = safe_cast(safe_cast(self.iterations.get(), float, 0), int, 0) # convert : string -> float -> int
    value = validValue(value, self.itMin, self.itMax)                     # get a valid value
    self.iterations.set(str(value))  # change text
    self.itScale.set(value)          # change scale

  # Scale event: Change the current value of the iterations
  def setIt(self, value):
    value = safe_cast(safe_cast(value, float, 0), int, 0) # convert : string -> float -> int
    value = validValue(value, self.itMin, self.itMax)     # get a valid value
    self.iterations.set(str(value))   # change text

  # change max and min iterations
  def setItMax(self, itMax):
    self.itMax = itMax
    value = self.iterations.get()
    if(int(value) > self.itMax):
      value = self.itMax
    
    self.itRange = self.itMax - self.itMin
    self.itScale.configure(to=self.itMax)
    self.iterations.set(str(value))
    self.itScale.set(value)
    
  def setItMin(self, itMin):
    self.itMin = itMin
    value = self.iterations.get()
    if(int(value) < self.itMin):
      value = self.itMin

    self.itRange = self.itMax - self.itMin
    self.itScale.configure(from_=self.itMin)
    self.iterations.set(str(value))
    self.itScale.set(value)
 

  # C1 and C2 events
  # Entry event (text field)
  def setTextC1(self, event):
    value = safe_cast(self.c1.get(), float, 0.0)  # convert : string -> float
    value = validValue(value, 0.1, 1.0)           # get a valid value
    self.c1.set(str(round(value, 2))) # change text
    self.c1Scale.set(value*100)       # change scale

  # Scale event: Change the current value of c1
  def setC1(self, value):
    value = safe_cast(value, float, 0.0)    # convert : string -> float
    value = validValue(value/100, 0.1, 1.0) # get a valid value
    self.c1.set(str(round(value, 2))) # change text
    
  # Entry event (text field)
  def setTextC2(self, event):
    value = safe_cast(self.c2.get(), float, 0.0)  # convert : string -> float
    value = validValue(value, 0.1, 1.0)           # get a valid value
    self.c2.set(str(round(value, 2))) # change text
    self.c2Scale.set(value*100)       # change scale

  # Scale event: Change the current value of c2
  def setC2(self, value):
    value = safe_cast(value, float, 0.0)    # convert : string -> float
    value = validValue(value/100, 0.1, 1.0) # get a valid value
    self.c2.set(str(round(value, 2))) # change text


  # Path-cost functions events
  # Entry event (text field)
  def setTextFunction(self, event):
    self.setFunction(self.function.get())
    return

  # Scale event: Change the current value of function
  def setFunction(self, value):
    value = safe_cast(value, int, 0)
    if(value < 0):
      self.function.set("1")
    elif(value > 5):
      self.function.set("5")
    else:
      self.function.set(str(value))
    return


  # change the weight function for graph creation
  def weightFunctionChange(self):
    if(self.appImage is not None):
      if(self.weightFunctionCombobox.get() == 'L0'):
        weightFunction = hg.WeightFunction.L0
      elif(self.weightFunctionCombobox.get() == 'L1'):
        weightFunction = hg.WeightFunction.L1
      elif(self.weightFunctionCombobox.get() == 'L2'):
        weightFunction = hg.WeightFunction.L2
      elif(self.weightFunctionCombobox.get() == 'L2_squared'):
        weightFunction = hg.WeightFunction.L2_squared
      elif(self.weightFunctionCombobox.get() == 'L_infinity'):
        weightFunction = hg.WeightFunction.L_infinity
      elif(self.weightFunctionCombobox.get() == 'max'):
        weightFunction = hg.WeightFunction.max
      elif(self.weightFunctionCombobox.get() == 'min'):
        weightFunction = hg.WeightFunction.min
      else:
        weightFunction = hg.WeightFunction.mean
    
      self.edge_weights = hg.weight_graph(self.graph, self.gradient_image, weightFunction)
      self.wsAttributeChange()

  # combobox event: change the weight function for graph creation
  def weightFunctionChangeEvent(self, event):
    self.weightFunctionChange()


  # change the watershed attribute
  def wsAttributeChange(self):
    if(self.appImage is not None):
      if(self.wsAttributeCombobox.get() == 'Area'):
        self.tree, self.altitudes = hg.watershed_hierarchy_by_area(self.graph, self.edge_weights)
      elif(self.wsAttributeCombobox.get() == 'Volume'):
        self.tree, self.altitudes = hg.watershed_hierarchy_by_volume(self.graph, self.edge_weights)
      else:
        self.tree, self.altitudes = hg.watershed_hierarchy_by_dynamics(self.graph, self.edge_weights)
    
    
  # combobox event: change the watershed attribute
  def wsAttributeChangeEvent(self, event):
    self.wsAttributeChange()


  def on_tab_selected(self, event):
    self.tab_text = event.widget.tab(event.widget.select(), "text")
    if(self.tab_text == 'Watershed' and self.appImage is not None):
      self.updateWatershedStructures()


  def updateWatershedStructures(self):
    if(self.tab_text == 'Watershed' and self.appImage is not None):
      self.weightFunctionChange()
      self.wsAttributeChange()

      # we construct a sketch of the saliency map just for illustration
      sm = hg.graph_4_adjacency_2_khalimsky(self.graph, hg.saliency(self.tree, self.altitudes))**0.5
      sm = sm[1::2,1::2]
      sm = np.pad(sm, ((0,1),(0,1)), mode='edge')
      sm = 1 - sm / np.max(sm)
      sm = np.dstack([sm]*3)
      sm = sm * 255
      self.sm = sm.astype(np.uint8)

  

#########################################################################
#     Image Aplication class
#########################################################################
class AppImage(DefaultImageWindow):
  def __init__(self, root, settingsPage, *args, **kwargs):
    ##-----------------------------------------
    ## call init method of child class
    DefaultImageWindow.__init__(self, root, 'Original image', settingsPage, *args, **kwargs)
    ##-----------------------------------------

    #self.appSettings = settingsPage
    
    # initialize variables
    self.imageNames = [] # image names to segmentation function
    self.images = [] # images readed from the directory
    self.markers = [] # markers coords
    self.marker = [] # last marker
    self.markers_sizes = []  # marker sizes
    self.drawing = False # indicates if mouse was pressed and not released
    self.i = 0 # image and imageNames index
    self.listBoxShow = 1
    self.listBoxShow_var = tk.StringVar()
    self.listBoxShow_var.set("1")
    self.imageNamesListBox = tk.StringVar()

    self.bgColor = (255,0,0)
    self.fgColor = (0,0,255)

    self.gradient_list = []
    self.graph_list = []

    self.canvas_rect = None
    self.canvas_lines = []
    self.start_x = None
    self.start_y = None

    self.canvas_shadow = None
    
    if(self.appSettings.scribblesType.get() == 0):
      self.color = self.fgColor
    else:
      self.color = self.bgColor
    self.objMarkers = 0

    ###########################################################################
    # tk objects
    self.viewMenu = tk.Menu(self.menu, tearoff=False)
    self.viewMenu.add_checkbutton(label="View images list", variable=self.listBoxShow_var, command=self.listBoxVisibility)
    self.menu.add_cascade(label="View", menu=self.viewMenu)

    #self.menu.add_command(label="Save", command=self.saveImage)
    self.saveMenu = tk.Menu(self.menu, tearoff=False)
    self.saveMenu.add_command(label="Save Image", command=self.saveImage)
    self.saveMenu.add_command(label="Save Scribbles", command=self.saveScribbles)
    self.menu.add_cascade(label="Save", menu=self.saveMenu)
    
    self.loadMenu = tk.Menu(self.menu, tearoff=False)
    self.loadMenu.add_command(label="Load Scribbles", command=self.loadScribbles)
    self.menu.add_cascade(label="Load", menu=self.loadMenu)

    self.root.bind("<KeyRelease-c>", self.clearAndUpdateImage) # c key release

    self.canvas.bind("<Motion>", self.mouseMotion)
    self.canvas.bind("<Button-1>", self.mousePressed)
    self.canvas.bind("<B1-Motion>", self.mouseMovementPressed) # mouse movement with the left button pressed 
    self.canvas.bind("<ButtonRelease-1>", self.mouseButtonReleased) # mouse button released

    self.canvas.bind("<Control-ButtonPress-1>", self.ctrl_mousePressed)
    self.canvas.bind("<Control-B1-Motion>", self.ctrl_mouseMovementPressed)
    self.canvas.bind("<Control-ButtonRelease-1>", self.ctrl_mouseButtonReleased)
    
    # add a listBox to show loaded images names and for choosen
    # scrolling the listbox
    self.listLabel = tk.Label(master=self.frame)
    self.listLabel.grid(row=0, column=1, columnspan=1, sticky=self.fill, padx=2, pady=2)
    self.createWidgets(self.listLabel)

    self.imagesListBox = tk.Listbox(master=self.listLabel, listvariable=self.imageNamesListBox, selectmode=tk.SINGLE)
    self.imagesListBox.grid(row=0, column=0, sticky='ns')
    self.createWidgets(self.imagesListBox)

    self.scrollbarList = AutoScrollbar(self.listLabel, orient='vertical')
    self.scrollbarList.grid(row=0, column=1, sticky='ns')
    
    self.imagesListBox.config(yscrollcommand=self.scrollbarList.set)
    self.scrollbarList.config(command=self.imagesListBox.yview)

    self.root.bind("<KeyRelease-Up>", self.changeImagePrevious) # mouse button released
    self.root.bind("<KeyRelease-Down>", self.changeImageNext) # mouse button released

    self.imagesListBox.bind("<ButtonRelease>", self.changeImageCursor) # mouse button pressed
    self.imagesListBox.bind("<KeyRelease-KP_Up>", self.changeImagePrevious) # mouse button released
    self.imagesListBox.bind("<KeyRelease-KP_Down>", self.changeImageNext) # mouse button released

    # <KeyPress-H> The user pressed the H key
    # KeyRelease The user let up on a key.
    # KP_Left 83 65430 ← on the keypad
    # KP_Right 85 65432 → on the keypad
    # KP_Up 80 65431 ↑ on the keypad
    # KP_Down 88 65433 ↓ on the keypad
    # Enter The user moved the mouse pointer into a visible part of a widget. (This is
    #       different than the enterkey,which is a KeyPress event for akeywhose name
    #       is actually 'return'.)
    # "<B1-Motion>" With the left mouse button held down, the mouse changed location.
    ###########################################################################


  # change the current image showed in window, 
  # clear all markers and change the max grid seeds in window settings accordingly with the new image
  def changeImage(self, i):
    self.imagesListBox.selection_clear(self.i)
    self.i = i
    self.imagesListBox.selection_set(self.i)
    self.imagesListBox.see(self.i)
    self.clearMarkers()
    self.clearImage()
    self.scale_img = 1.0
    self.image2Tk()
    self.appSettings.destroyDefaultWindows()
    self.clearMarkers()
    self.updateWatershedStructures()
    
  def updateWatershedStructures(self):
    self.appSettings.gradient_image = self.gradient_list[self.i]
    self.appSettings.graph = self.graph_list[self.i]
    self.appSettings.updateWatershedStructures()

  def listBoxVisibility(self):
    if(self.listBoxShow == 1):
      self.listLabel.grid_forget()
      self.listBoxShow = 0
    else:
      self.listLabel.grid(row=0, column=1, columnspan=1, sticky=tk.N+tk.S+tk.E+tk.W, padx=5, pady=5)
      self.createWidgets(self.listLabel)
      self.listBoxShow = 1

  # ListBox left click released event: change the current image
  def changeImageCursor(self, event): # event button select clicked
    pos = self.imagesListBox.curselection()
    if(len(pos) > 0 and self.i != pos[0]):
      self.changeImage(pos[0])
      
  # UP key released event: change the current image
  def changeImagePrevious(self, event):
    pos = self.imagesListBox.curselection()
    if(len(pos) > 0 and pos[0] == self.i and self.i > 0):
      self.changeImage(self.i-1)
  
  # DOWN key released event: change the current image
  def changeImageNext(self, event):
    pos = self.imagesListBox.curselection()
    if(len(pos) > 0 and pos[0] == self.i and self.i < len(self.images)-1):
      self.changeImage(self.i+1)

  # c key released event: clear the markers and make a copy from the original image to the current image
  def clearAndUpdateImage(self, event):
    self.clearMarkers()
    self.clearImage()
    self.image2Tk()
    
  
  # clear the markers and updated it in settings window
  def clearMarkers(self):
    width, height = self.currentImage.size
    self.img_markers = Image.new('RGB', (width, height), color = (0,0,0))


  # buttom event: delete the object markers (settings window)
  def clearObjMarkers(self):
    img_markers = np.array(self.img_markers)
    img_markers[:,:,2] = img_markers[:,:,2] * 0
    self.img_markers = Image.fromarray(img_markers)
    self.clearImage()
    self.drawMarkers()
    self.image2Tk()  


  # buttom event: delete the object markers (settings window)
  def clearBgMarkers(self):
    img_markers = np.array(self.img_markers)
    img_markers[:,:,0] = img_markers[:,:,0] * 0
    self.img_markers = Image.fromarray(img_markers)
    self.clearImage()
    self.drawMarkers()
    self.image2Tk()  


  # make a copy from the original image to the current image
  def clearImage(self):
    self.currentImage = self.images[self.i].copy()

  # since the image cavas can be resized by zoom function, 
  # and the image can has a bigger size than the window size, 
  # we need to recalculate its coordinates
  def getImgCoords(self, x, y):
    # get canvas position considering the scrollbars
    x = self.canvas.canvasx(x)
    y = self.canvas.canvasy(y)

    # remove image scale from position
    x = int(x / self.scale_img)
    y = int(y / self.scale_img)
    
    return x,y

  # convert an 8bits rgb color to an hex #rrggbb form
  def rgbToHex(self, color):
    # convert 3x8 bits color to hex color #rrggbb -> #000000 (black) to #ffffff (white)
    hexColor = str(hex(color[2])[2:])
    if(len(str(hex(color[2])[2:])) == 1):
      hexColor = "0" + hexColor

    hexColor = str(hex(color[1])[2:]) + hexColor
    if(len(str(hex(color[1])[2:])) == 1):
      hexColor = "0" + hexColor

    hexColor = str(hex(color[0])[2:]) + hexColor
    if(len(str(hex(color[0])[2:])) == 1):
      hexColor = "0" + hexColor
    hexColor = "#" + hexColor
    return hexColor
  
  def ctrl_mousePressed(self, event):
    if(self.appSettings.scribblesType.get() != 0 and self.appSettings.scribblesType.get() != 1):
      return
    
    if(self.appSettings.scribblesType.get() == 0):  self.color = self.fgColor
    else: self.color = self.bgColor

    if(self.canvas_rect is not None or len(self.canvas_lines) > 0):
      return

    color = self.rgbToHex(self.color)

    # save mouse drag start position
    self.start_x = int(self.canvas.canvasx(event.x))
    self.start_y = int(self.canvas.canvasy(event.y))

    width = self.appSettings.scribblesSizeScale.get()-1
    
    # create a rectangle
    self.canvas_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, fill="", width=int(width*self.scale_img), outline=color)
    self.canvas_center_rect = None
  

  def ctrl_mouseMovementPressed(self, event):
    if(self.appSettings.scribblesType.get() != 0 and self.appSettings.scribblesType.get() != 1):
      return

    if(len(self.canvas_lines) > 0 or self.canvas_rect is None):
      return

    # get canvas position considering the scrollbars
    curX = int(self.canvas.canvasx(event.x))
    curY = int(self.canvas.canvasy(event.y))

    # expand rectangle as you drag the mouse
    self.canvas.coords(self.canvas_rect, self.start_x, self.start_y, curX, curY)
    
    if(self.canvas_center_rect is not None):
      self.canvas.delete(self.canvas_center_rect)

    width = self.appSettings.scribblesSizeScale.get()*self.scale_img
    if(abs(self.start_x-curX) > width*2.5 and abs(self.start_y-curY) > width*2.5):
      center = (int(min(self.start_x,curX) + abs(self.start_x-curX)/2), int(min(self.start_y,curY) + abs(self.start_y-curY)/2))
      if(self.color == self.fgColor): 
        color = self.rgbToHex(self.bgColor)
      else: 
        color = self.rgbToHex(self.fgColor)
      self.canvas_center_rect = self.canvas.create_oval((center[0]-int(width/2)), (center[1]-int(width/2)), (center[0]+int(width/2)), (center[1]+int(width/2)), fill=color, width=0) 


  def ctrl_mouseButtonReleased(self, event):
    if(self.appSettings.scribblesType.get() != 0 and self.appSettings.scribblesType.get() != 1):
      return
          
    exit = False
    for line in self.canvas_lines:
      self.canvas.delete(line)
      exit = True
    self.canvas_lines = []
    if exit: return

    # get end position
    end_x, end_y = self.getImgCoords(event.x,event.y)

    width = self.appSettings.scribblesSizeScale.get()-1

    # remove image scale from initial position
    self.start_x,self.start_y = (int(self.start_x / self.scale_img), int(self.start_y / self.scale_img))

    # tkinter and pillow draws a rectangle differently
    self.start_x -= int(width/2)
    end_x += int(width/2)
      
    if(self.start_y < end_y): 
      self.start_y -= int(width/2)
      end_y += int(width/2)
    else:
      self.start_y += int(width/2)
      end_y -= int(width/2)   

    # draw a rectangle using pillow
    draw = ImageDraw.Draw(self.img_markers)
    
    draw.rectangle([self.start_x, self.start_y, end_x, end_y], outline=self.color, width=int(width))

    if(abs(self.start_x-end_x) > width*2.5 and abs(self.start_y-end_y) > width*2.5):
      center = (min(self.start_x,end_x) + abs(self.start_x-end_x)/2, min(self.start_y,end_y) + abs(self.start_y-end_y)/2)
      if(self.color == self.fgColor): 
        color = self.bgColor
      else: 
        color = self.fgColor

      draw.ellipse((center[0]-int(width/2), center[1]-int(width/2), center[0]+int(width/2), center[1]+int(width/2)), fill=color, width=0)
      

    self.drawMarkers()
    self.image2Tk()


  def mouseMotion(self, event):
    if(self.canvas_rect is not None):
      return
    
    # save mouse drag start position
    x = int(self.canvas.canvasx(event.x))
    y = int(self.canvas.canvasy(event.y))

    if(self.canvas_shadow is not None):
      self.canvas.delete(self.canvas_shadow)

    radio = int((self.appSettings.scribblesSizeScale.get()*self.scale_img)/2)

    self.canvas_shadow = self.canvas.create_oval(x-radio, y-radio, x+radio, y+radio, fill='white', stipple='gray25', width = 0)



  def mousePressed(self, event):
    if(self.appSettings.scribblesType.get() == 0): self.color = self.fgColor
    else: self.color = self.bgColor

    if(self.canvas_shadow is not None):
      self.canvas.delete(self.canvas_shadow)

    self.lines = []

    if(self.canvas_rect is not None or len(self.canvas_lines) > 0):
      return

    # convert 3x8 bits color to hex color #rrggbb -> #000000 (black) to #ffffff (white)
    color = str(hex(self.color[2])[2:])
    if(len(str(hex(self.color[2])[2:])) == 1):
      color = "0" + color

    color = str(hex(self.color[1])[2:]) + color
    if(len(str(hex(self.color[1])[2:])) == 1):
      color = "0" + color

    color = str(hex(self.color[0])[2:]) + color
    if(len(str(hex(self.color[0])[2:])) == 1):
      color = "0" + color
    color = "#" + color
    self.hexColor = color

    # save mouse drag start position
    self.start_x = int(self.canvas.canvasx(event.x))
    self.start_y = int(self.canvas.canvasy(event.y))


  # left key pressed and moving event: draw marker
  def mouseMovementPressed(self, event):
    if(self.canvas_rect is not None):
      return
    elif(self.appSettings.scribblesType.get() == 3):
      x,y = self.getImgCoords(event.x,event.y)
      self.eraseScribbles(x,y)
      return    

    if(self.canvas_shadow is not None):
      self.canvas.delete(self.canvas_shadow)

    # get canvas position considering the scrollbars
    curX = int(self.canvas.canvasx(event.x))
    curY = int(self.canvas.canvasy(event.y))

    width = self.appSettings.scribblesSizeScale.get()

    canvas_line = self.canvas.create_line(self.start_x, self.start_y, curX, curY, fill=self.hexColor, width=int(width*self.scale_img), joinstyle='round', capstyle='round', smooth=True)
    self.canvas_lines.append(canvas_line)

    for (x,y) in [(self.start_x, self.start_y), (curX, curY)]:
      self.lines.append((int(x / self.scale_img),int(y / self.scale_img)))
    
    self.start_x = curX
    self.start_y = curY


  
  # left key released event: draw marker
  def mouseButtonReleased(self, event):
    
    if(self.canvas_rect is not None):
      self.canvas.delete(self.canvas_rect)
      self.canvas_rect = None
      if(self.canvas_center_rect is not None):
        self.canvas.delete(self.canvas_center_rect)
        self.canvas_center_rect = None
      return
    
    if(self.canvas_shadow is not None):
      self.canvas.delete(self.canvas_shadow)
    
    x, y = self.getImgCoords(event.x,event.y)

    if(self.appSettings.scribblesType.get() == 3):
      self.eraseScribbles(x, y)
      return

    ratio = int(self.appSettings.scribblesSizeScale.get()/2)-1
      
    draw = ImageDraw.Draw(self.img_markers)
      
    if(len(self.lines) > 0):
      start_x, start_y = self.lines[0]
      draw.ellipse((start_x-ratio, start_y-ratio, start_x+ratio, start_y+ratio), fill=self.color, width=0)
        
      for (x,y) in self.lines[1:]:
        #draw.line([(start_x,start_y), (x,y)], fill=self.color, width=ratio, joint='curve')
        line = self.bresenham_line(start_x, start_y, x, y)   
        line.pop(0)
        
        for [x2,y2] in line:
          draw.ellipse((x2-ratio, y2-ratio, x2+ratio, y2+ratio), fill=self.color, width=0)
          
        start_x, start_y = (x,y)
          
    else:
      draw.ellipse((x-ratio, y-ratio, x+ratio, y+ratio), fill=self.color, width=0)

    self.drawMarkers()
    self.image2Tk()


  def eraseScribbles(self, x, y):
    ratio = int(self.appSettings.scribblesSizeScale.get()/2)-1
    
    # draws a line using pillow
    draw = ImageDraw.Draw(self.img_markers)
    if(self.start_y is not None and self.start_x is not None):
      line = self.bresenham_line(self.start_x, self.start_y, x, y)  
      for [x2,y2] in line:
        draw.ellipse((x2-ratio, y2-ratio, x2+ratio, y2+ratio), fill=(0,0,0), width=0)
    else:
      draw.ellipse((x-ratio, y-ratio, x+ratio, y+ratio), fill=(0,0,0), width=0)

    self.start_x = x
    self.start_y = y

    self.clearImage()
    self.drawMarkers()
    self.image2Tk()

  
  def drawMarkers(self):
    img1 = np.array(self.getCurrentImage())
    if(len(img1.shape) == 2):
      img1 = np.stack((img1,)*3, axis=-1)
    img2 = np.array(self.img_markers)

    mask = np.bitwise_and(255-img2[:,:,0], 255-img2[:,:,2]) # 0 in markers coords
    
    # set 0 in red and blue markers in all channels
    img1[:,:,0] = np.bitwise_and(img1[:,:,0], mask)
    img1[:,:,1] = np.bitwise_and(img1[:,:,1], mask)
    img1[:,:,2] = np.bitwise_and(img1[:,:,2], mask)

    # set 255 in red and blue markers in its respective channels
    img1[:,:,0] = np.bitwise_or(img1[:,:,0], img2[:,:,0])
    img1[:,:,2] = np.bitwise_or(img1[:,:,2], img2[:,:,2])

    self.currentImage = Image.fromarray(img1) # convert from opencv to PIL


  # create a list of images, imageNames and the variable list for the ListBox
  # Call the destructor function if no image could be read.
  def setImages(self, imageNames):
    self.images = []
    self.imageNames = []
    i = 0

    for imgFile in imageNames:
      if(os.path.isfile(imgFile) is True):

        if(imgFile.split('.')[-1] == 'dcm'):
          dicomImg = DicomImage(imgFile)
          image = dicomImg.getPILImage()
          #image = dicomImg.getPILImage2()
        else:
          try:
            image = Image.open(imgFile)
          except Exception as inst:
            break

        if(image is not None):
          self.images.append(image)
          self.imageNames.append(imgFile)
          self.imagesListBox.insert(i, imgFile.split("/")[-1])
          i+=1

          image = np.array(image)
          image = image.astype(np.float32)/255
          if(len(image.shape) == 2):
            image = np.stack((image,)*3, axis=-1)

          self.gradient_list.append(self.appSettings.detector.detectEdges(image))
          self.graph_list.append(hg.get_4_adjacency_graph(image.shape[:2]))
          
    if(len(self.images) == 0):
      self.destructor()
      message = messagebox.showerror("Error", "Could not read any image.")
    else:
      self.imagesListBox.selection_set(self.i)
      self.clearImage()
      self.clearMarkers()
      self.image2Tk()
      self.updateWatershedStructures()
      self.scale_img = 1.0
  

  

  def saveImage(self):
    image = self.currentImage
    if(image is not None):
      folder_selected = filedialog.asksaveasfilename(initialdir = "./",title = "Save file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"),("all files","*.*")))
    if(len(folder_selected) != 0):
      image.save(folder_selected)
    else:
      message = messagebox.showerror("Error", "No image founded.")
      return


  # buttom menu event: Save scribbles to a text file
  def saveScribbles(self):
    self.getMarkers()

    if(len(self.markers) == 0): 
      message = messagebox.showerror("Error", "No scribbles founded.")
      return

    folder_selected = filedialog.asksaveasfilename(initialdir = "./",title = "Save file", filetypes = (("Text File", "*.txt"),("all files","*.*")))
    if(len(folder_selected) == 0):
      return
    
    extension = folder_selected.split(".")[-1]
    if(extension != 'txt'):
      message = messagebox.showerror("Error", "Invalid format. Please use .txt")
      return

    f = open(folder_selected, 'w')
    f.write("%d\n"%(len(self.markers_sizes)))
    f.write("%d\n"%(self.markers_sizes[0]))
    
    index_sizes=0
    acum=0
    
    for i in range(len(self.markers)-1):
      if(acum == self.markers_sizes[index_sizes]):
        index_sizes+=1
        acum=0
        f.write("%d\n"%(self.markers_sizes[index_sizes]))
      
      [x,y] = self.markers[i]
      f.write("%d;%d\n"%(x,y))
      acum+=1

    if(acum == self.markers_sizes[index_sizes]):
      index_sizes+=1
      acum=0
      f.write("%d\n"%(self.markers_sizes[index_sizes]))
    
    [x,y] = self.markers[-1]
    f.write("%d;%d\n"%(x,y))
    f.write("%d"%(self.objMarkers))
    f.close()


  # from points p0 and p1, return a line (list of points) from p0 to p1
  def bresenham_line(self, x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
      x0, y0 = y0, x0  
      x1, y1 = y1, x1

    switched = False
    if x0 > x1:
      switched = True
      x0, x1 = x1, x0
      y0, y1 = y1, y0

    if y0 < y1: 
      ystep = 1
    else:
      ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []    
    for x in range(x0, x1 + 1):
      if steep:
          line.append([y,x])
      else:
          line.append([x,y])

      error = error + deltay
      if error > 0:
        y = y + ystep
        error = error - deltax
    if switched:
      line.reverse()
    return line

  # Remove duplicate points in sequence and adds points to connect all coordinates in the sequence entry marker. Returns a connected marker
  # It does not check for non-sequential duplicate points.
  def getConnectedMarker(self, marker):
    newMarker = []
    x_ant = -1
    y_ant = -1
    for [x,y] in marker:
      # if its not the initial point and is disconnected to the previous point
      if((x_ant != -1 and y_ant != -1) and (abs(x_ant - x) >= 1 or abs(y_ant - y) >= 1)):
        line = self.bresenham_line(x_ant, y_ant, x, y)
        line.pop(0)
        newMarker.extend(line)
      else:
        if(x_ant != x or y_ant != y):
          newMarker.append([x, y])
      x_ant, y_ant = newMarker[-1]
    
    return newMarker

  # buttom menu event: Load scribbles from a text file
  def loadScribbles(self):
    folder_selected = filedialog.askopenfilename(initialdir = "./",title = "Select file")
    if(len(folder_selected) == 0):
      return

    if(folder_selected.split('.')[-1] != 'txt'):
      message = messagebox.showerror("Error", "Invalid format.")
      return

    # lê o arquivo txt e armazena as coordenadas
    f = open(folder_selected, 'r')
    lines = f.readlines()

    coords = []
    size_scribbles = []
    size_scribbles.append(int(lines[1]))
    max_scribbles = int(lines[0])
    
    index_sizes = 0
    acum=0

    for i in range(2,len(lines)):
      if(acum == size_scribbles[index_sizes] and index_sizes < max_scribbles-1):
        acum=0
        index_sizes+=1
        size_scribbles.append(int(lines[i]))
      else:
        coords.append(tuple([int(n) for n in lines[i].split(';')]))
        acum+=1

    if(len(size_scribbles) == 0):
      return

    if(len(coords[-1]) == 1): 
      num_objects = coords[-1][0]
      coords = coords[:-1] # last value isn't a coordinate
    else:
      num_objects = 1
    
    # define o numero de scribbles de objeto
    popup = PopupSpinbox(tk.Toplevel(self.root), 'Number of object scribbles', 'How many object scribbles?', num_objects, len(size_scribbles))
    self.root.wait_window(popup.root)
    
    if(popup.setAnswer is False):
      return

    # desenha os scribbles
    self.clearMarkers()
    self.clearImage()
    draw = ImageDraw.Draw(self.img_markers)
    num_objects = popup.intAnswer
    
    obj_coords = 0
    for size in size_scribbles[:num_objects]:
      obj_coords += size

    draw.point(coords[:obj_coords], fill=self.fgColor)
    draw.point(coords[obj_coords:], fill=self.bgColor)
    
    self.drawMarkers() # draw the image scribbles in the image showed
    self.image2Tk()


  # return the markers
  def getMarkers(self):
    self.markers = []
    self.objMarkers = 0
    self.markers_sizes = []
    
    image, number_of_objects = ndimage.label(np.array(self.img_markers)[:,:,2])
    blobs = ndimage.find_objects(image)
    
    for i,j in enumerate(blobs):
      marker = []
      for y in range(j[0].start,j[0].stop):
        for x in range(j[1].start,j[1].stop):
          if(image[y,x] != 0):
            marker.append([x,y])
      
      self.markers_sizes.insert(0, len(marker))
      self.markers = marker + self.markers
      self.objMarkers += 1
    
    image, number_of_objects = ndimage.label(np.array(self.img_markers)[:,:,0])
    blobs = ndimage.find_objects(image)
    
    for i,j in enumerate(blobs):
      marker = []
      for y in range(j[0].start,j[0].stop):
        for x in range(j[1].start,j[1].stop):
          if(image[y,x] != 0):
            marker.append([x,y])
      
      self.markers_sizes.append(len(marker))
      self.markers = self.markers + marker
    
    return self.markers


  # return a list of the number os pixels in each scribble
  def getMarkerSizes(self):
    return self.markers_sizes


  # return the current image name
  def getImageName(self):
    return self.imageNames[self.i]

  # return the curent image
  def getCurrentImage(self):
    image = self.images[self.i]
    return image


#########################################################################
#     Segmentation Window class
#########################################################################
class SegmentationWindow(DefaultImageWindow):
  def __init__(self, root, titlePage, settingsPage, *args, **kwargs):
    ##-----------------------------------------
    ## call init method of child class
    DefaultImageWindow.__init__(self, root, titlePage, settingsPage, *args, **kwargs)
    ##-----------------------------------------

    # variables
    self.blueMask = tk.IntVar()
    self.blueMask.set(1)

    self.redMask = tk.IntVar()
    self.redMask.set(1)

    self.img_fg = None
    self.img_bg = None
    self.orig_img = None
    self.imageName = None

    # checkboxes
    self.masksLabelFrame = tk.LabelFrame(master=self.frame, text='Masks')
    self.masksLabelFrame.grid(row=1, column=0, columnspan=1, sticky=self.fill, padx=6, pady=4)
    self.createWidgets(self.masksLabelFrame)

    self.checkMask1 = tk.Checkbutton(master=self.masksLabelFrame, text='Blue (object)', command=self.setBlueMask, variable=self.blueMask)
    self.checkMask1.grid(row=0, column=0, columnspan=1, sticky=tk.W, padx=0, pady=1)
    self.createWidgets(self.checkMask1)

    self.checkMask2 = tk.Checkbutton(master=self.masksLabelFrame, text='Red (background)', command=self.setRedMask, variable=self.redMask)
    self.checkMask2.grid(row=0, column=1, columnspan=1, sticky=tk.W, padx=0, pady=1)
    self.createWidgets(self.checkMask2)

  # override parent class method
  def setImage(self, segm_img, img_fg, img_bg, orig_img, imageName):
    self.imageName = imageName
    self.img_fg = img_fg
    self.img_bg = img_bg
    self.orig_img = orig_img # RGB image
    self.segm_img = segm_img
    
    if(self.img_fg is None or self.img_bg is None or self.orig_img is None):
      self.destructor()
    else:
      self.currentImage = segm_img
      self.image2Tk()


  def setBlueMask(self):
    if(self.blueMask.get() == 1 and self.redMask.get() == 1):
      self.currentImage = self.segm_img
    elif(self.blueMask.get() == 1 and self.redMask.get() == 0):
      self.currentImage = self.img_fg
    elif(self.blueMask.get() == 0 and self.redMask.get() == 1):
      self.currentImage = self.img_bg
    else:
      self.currentImage = self.orig_img
    
    self.image2Tk()


  def setRedMask(self):
    if(self.blueMask.get() == 1 and self.redMask.get() == 1):
      self.currentImage = self.segm_img
    elif(self.blueMask.get() == 1 and self.redMask.get() == 0):
      self.currentImage = self.img_fg
    elif(self.blueMask.get() == 0 and self.redMask.get() == 1):
      self.currentImage = self.img_bg
    else:
      self.currentImage = self.orig_img
    
    self.image2Tk()


def main(): 
  root = tk.Tk()
  app = SettingsWindow(root)
  root.mainloop()

if __name__ == '__main__':
  main()

