
# coding: utf-8

# In[ ]:


# Exposure Splicing
# Lily Kuntz
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
import PIL.Image, PIL.ImageTk
import skimage.io as skio
import skimage.color as skc
import skimage.morphology as skm
import skimage.measure as skme
import numpy as np
import scipy.ndimage.filters as sknf
import skimage.exposure as ske
import skimage.filters as skf

def main(imgAnc, imgBnc, thrsA, runAlignment, onlyLgst, horizonBlur):
    # this is the main function that takes the two original (color)
    # images as input, and returns the final image combination
    #
    # image A is for the image you want to keep the darker portion of
    # image B is for the image you want to keep the lighter portion of
    # onlyLgst is a boolean to signal if only the largest connected component should be selected, 
    # or if every component above the given threshold ought to be used (imgA thrs. used only)
    
    if (imgAnc.shape != imgBnc.shape):
        print("Please select two images of equal dimensions for input.")
        return
    
    
    if (runAlignment):
        imgA, imgB = alignment(imgAnc, imgBnc)     # nc - not (yet) cropped original images
    else:
        imgA = imgAnc
        imgB = imgBnc
        
    
    res1, maskA = combine(imgA, imgB, thrsA, onlyLgst)
    
    
    # now done creating our new image, except for blurring the horizon line a little bit
    # so that the two disparate parts look a bit more cohesive...
    
    if (horizonBlur):
        lineBlurred2, split2 = blurHorizon(res1, maskA)
        # to join the two final pieces together and display
        res2 = lineBlurred2 + split2
    else:
        res2 = res1 
        
    #resAlmostFinal = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
    resFinal = ske.rescale_intensity(res2)
    
    # modify these lines for the user interface
    if (runAlignment):
        resFinal = display(resFinal, False)     # because the runAlignment boolean has converted the colors to RGB already
    else:
        resFinal = display(resFinal, True)     # because the colors still need to be converted back to RGB
    
    cv2.imwrite("finalimg.jpg", resFinal)

def alignment(imgA, imgB):
    # takes two images as input, aligns them, cropps the larger one to the size of the
    # smaller one, then returns the two final images, image A and image B
    
    # color channels for image B
    imgBRed = imgB[:,:,0]
    imgBGreen = imgB[:,:,1]
    imgBBlue = imgB[:,:,2]
    
    # color channels for image A
    imgARed = imgA[:,:,0]
    imgAGreen = imgA[:,:,1]
    imgABlue = imgA[:,:,2]
    
    # convert images to grayscale (for correct sizing without color channels)
    imgA_gray = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    
    # find size of imgA; used when to ensure imgB will be same size as image 1 when aligned to it
    size = imgA_gray.shape
    
    # define Affine motion model for alignment
    align_mode = cv2.MOTION_AFFINE
    # define and initialize 2x3 matrices
    align_matrix = np.eye(2, 3, dtype=np.float32)
    
    # specify the number of iterations
    number_of_iterations = 1000
    # specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    # define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    
    # run the ECC algorithm for each color channel, results are stored in align_matrix.
    (ccR, align_matrixR) = cv2.findTransformECC(imgA_gray,imgBRed,align_matrix, align_mode, criteria)
    (ccG, align_matrixG) = cv2.findTransformECC(imgA_gray,imgBGreen,align_matrix, align_mode, criteria)
    (ccB, align_matrixB) = cv2.findTransformECC(imgA_gray,imgBBlue,align_matrix, align_mode, criteria)
    
    # use warpAffine for Affine transformation
    imgB_alignedRed = cv2.warpAffine(imgBRed, align_matrixR, (size[1],size[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    imgB_alignedGreen = cv2.warpAffine(imgBGreen, align_matrixG, (size[1],size[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    imgB_alignedBlue = cv2.warpAffine(imgBBlue, align_matrixB, (size[1],size[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    # final outputs
    imgB_final = cv2.merge((imgB_alignedBlue, imgB_alignedGreen, imgB_alignedRed))
    imgA_final = cv2.merge((imgABlue, imgAGreen, imgARed)) # necessary for showing "real colors" image
    
    cv2.imwrite("Final_Algined.jpg", imgB_final)
    imgB_final = cv2.imread("Final_Algined.jpg")
    
    # and now to crop them, then return them
    imgA_croppedFinal, imgB_croppedFinal = crop(imgA_final, imgB_final)
    
    return(imgA_croppedFinal, imgB_croppedFinal)

def crop(imgA_final, imgB_final):
    # cropping imgB (b) in reference to how it shifted
    # cropping imgA (a) in reference to how imgB shifted
    i = 0
    j = 0
    l = imgB_final.shape[0] - 1
    m = imgB_final.shape[1] - 1
    
    
    while ((imgB_final[i,j,0] == 0) & (imgB_final[i,j,1] == 0) & (imgB_final[i,j,2] == 0)):
        i = i+25
        j = j+25
        
    while ((imgB_final[l,m,0] == 0) & (imgB_final[l,m,1] == 0) & (imgB_final[l,m,2] == 0)):
        l = l-25
        m = m-25
        
        
    final_imgA1 = imgA_final[:,j:m, ...]
    cropped_imgA = final_imgA1[i:l,:, ...]
    final_imgB1 = imgB_final[:,j:m, ...]
    cropped_imgB = final_imgB1[i:l,:, ...]
    
    
    return(cropped_imgA, cropped_imgB)

def combine(imgA, imgB, thrsA, onlyLgst):
    # takes the two initial images as input and returns the conjoined
    # version without the horizon blur, along with maskA
    
    bwA = skc.rgb2gray(imgA)
    bwB = skc.rgb2gray(imgB)
    

    # this series of connected componenets and inversions is meant to remove both
    # extraneous black and white spots from each mask (maskA and maskB)
    imgAThrs = bwA > thrsA
    imgBThrs = ~imgAThrs     # B's thresholded image is solely dependent on what remains after A's
    
    if (onlyLgst):
    
        maskA1 = lgstComp(imgAThrs)
        maskA2 = ~maskA1
        maskA3 = lgstComp(maskA2)
        maskA = ~maskA3
    
        maskB = maskA3
        # mask B does not need to be inverted, because we want the foreground below mask A
    
    else:
        maskA = imgAThrs
        maskB = ~imgAThrs     # notice that image A's threshold is used in this case
    

    # alignment would happen here
    
    imgMskA = imgA.copy()
    imgMskB = imgB.copy()

    imgMskA[maskA] = 0
    imgMskB[maskB] = 0    # already 'left inverted' above when thresholding
    

    #these three comment lines below can be deleted, it's just another way to merge the two pieces:
    #masks1 = [imgMskA, imgMskB]
    #mergeMertens1 = cv2.createMergeMertens()
    #res1 = mergeMertens1.process(masks1)
    
    res1 = imgMskA | imgMskB

    
    return res1, maskA

def lgstComp(imgT):
    # returns the mask of the largest component (true) and the rest of the image (false)
    
    lblGero = skm.label(imgT, neighbors = 8)
    compGero = skme.regionprops(lblGero)
    
    maxGero = 0
    idxGero = None

    for i in compGero:
        if (i.area > maxGero):
            maxGero = i.area
            idxGero = i.label
            
    lbl = skm.label(imgT, neighbors = 8)
    mask = (lbl == idxGero)
    
    return mask

def blurHorizon(res1, maskA):
    # takes as input maskA and res1 and returns the two partial
    # images, to be comjoined within main() into the final image
    
    strel = skm.rectangle(5, 5)
    
    # now to manipulate the mask A until we have just the horizon line left
    lineAdil = skm.dilation(maskA, selem = strel)
    horizonMaskMessy = lineAdil ^ maskA     # because mask B was created from mask A anyway
    horizonMask = lgstComp(horizonMaskMessy)
    
    #these three comment lines below can be deleted, it's just another way to merge the two pieces:
    #lineAerr = skm.erosion(maskA, selem = strel)    # because mask B was created from mask A anyway
    #horizonMaskMessyB = lineAerr ^ maskA
    #horizonMaskMessy = horizonMaskMessyA | horizonMaskMessyB

    
    line = res1.copy()     # line will - eventually - have the masked horizon line only (with the median filter)
    split = res1.copy()    # split will contain the rest of the image "res1" (without the median filter)
    
    lineBlurred = sknf.median_filter(line, footprint=np.ones((4, 4, 3)))
    
    lineBlurred[~horizonMask] = 0
    split[horizonMask] = 0
    
    
    # now to convert it to a uniform format so that both images line up
    
    split2 = ske.rescale_intensity(split)
    lineBlurred2 = ske.rescale_intensity(lineBlurred)
    
    
    return lineBlurred2, split2

def display(img, toBeConverted):
    # this is just for the diplay, and you're free to delete it
    # if you display the images differently in the interface
    
    if (toBeConverted):
        imgRet = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imgRet
        #plt.figure(figsize = (11, 10))
        #skio.imshow(imgRet, cmap = 'gray')
    else:
        return img
        #plt.figure(figsize = (11, 10))
        #skio.imshow(img, cmap = 'gray')
        
class Window(Frame):
    
    def __init__(self, master = None):
        Frame.__init__(self, master)
        
        self.master = master
        
        self.init_window()
    
    def init_window(self):
        self.configure(background = "black")
        
        self.master.title("Exposure Splicing Project")
        
        self.pack(fill = BOTH, expand = 1)
        
        menu = Menu(self.master)
        self.master.config(menu = menu)
        
        file = Menu(menu)
        file.add_command(label = 'Upload Image 1', command = self.upload1)
        file.add_command(label = 'Upload Image 2', command = self.upload2)
        menu.add_cascade(label = 'File', menu = file)
        
        self.slider1 = Scale(troughcolor = "orange", from_=0, to=1, resolution = 0.01, orient="horizontal", label = "Threshold", bg = "white", length = 200, font=("Helvetica", 16))
        self.slider1.bind("<ButtonRelease-1>", self.fix)
        self.slider1.place(x = 450, y = 200)
        
        self.var1 = IntVar()
        self.onlyLargest = Checkbutton(text="I want only the largest piece of image 1 above my chosen threshold.", variable= self.var1, bg = "black", fg = "white", font=("Helvetica", 16))
        self.onlyLargest.place(x = 700, y = 205)
        self.onlyLargest.v = self.var1
        
        self.var2 = BooleanVar()
        self.blur = Checkbutton(text="I want to blur just a thin sliver of my final image along the intersection between the two original images.", variable= self.var2, bg = "black", fg = "white", font=("Helvetica", 16))
        self.blur.place(x = 700, y = 230)
        self.blur.v = self.var2
        
        self.var3 = BooleanVar()
        self.align = Checkbutton(text="I want to align the imges first.", variable= self.var3, bg = "black", fg = "white", font=("Helvetica", 16))
        self.align.place(x = 700, y = 180)
        self.align.v = self.var3
        
        self.b = Checkbutton(variable=0, selectcolor = "orange", indicatoron=0, text="Fix", background = "black", foreground= "white", font= ("Helvetica", 16))
        self.b.bind("<ButtonRelease-1>", self.fix)
        self.b.deselect()
        self.b.place(x = 700, y = 250)

    def showImg1(self): 
        load = Image.open("firstimg.jpg")
        basewidth = 400
        wpercent = (basewidth / float(load.size[0]))
        hsize = int((float(load.size[1]) * float(wpercent)))
        load = load.resize((basewidth, hsize), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image = render)
        img.image = render
        img.place(x= 15, y = 200)

    def upload1(self):
        file = cv2.imread(filedialog.askopenfilename())
        cv2.imwrite("firstimg.jpg", file)
        self.showImg1()
    
    def showImg2(self): 
        load = Image.open("secondimg.jpg")
        basewidth = 400
        wpercent = (basewidth / float(load.size[0]))
        hsize = int((float(load.size[1]) * float(wpercent)))
        load = load.resize((basewidth, hsize), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image = render)
        img.image = render
        img.place(x=1000, y=200)

    def upload2(self):
        file = cv2.imread(filedialog.askopenfilename())
        cv2.imwrite("secondimg.jpg", file)
        self.showImg2()
    
    def showImg3(self): 
        load = Image.open("finalimg.jpg")
        basewidth = 550
        wpercent = (basewidth / float(load.size[0]))
        hsize = int((float(load.size[1]) * float(wpercent)))
        load = load.resize((basewidth, hsize), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image = render)
        img.image = render
        img.place(x=435, y=150)
        
    def fix(self, event):
        self.b.deselect()
        imgA = skio.imread("firstimg.jpg")
        imgB = skio.imread("secondimg.jpg")
        thrsA = self.slider1.get() 
        align = self.var3.get()
        onlyLgst = self.var1.get()
        blur = self.var2.get()
        
        main(imgA, imgB, thrsA, align, onlyLgst, blur)
        
        self.showImg3()

root = Tk()
root.geometry("1500x1000")
root.configure(background = "black")

instruction1 = Label(text = "Step 1: File - upload img 1", font=("Helvetica", 18, 'bold'), bg = "black", fg = "white")
instruction1.pack(anchor = CENTER, pady = 10)

instruction2 = Label(text = "Step 2: File - upload img 2",  font=("Helvetica", 18, 'bold'), bg = "black", fg = "white")
instruction2.pack(anchor = CENTER, pady= 5)

instruction3 = Label(text = "Step 3: Adjust threshold value", font=("Helvetica", 18, 'bold'), bg = "black", fg = "white")
instruction3.pack(anchor = CENTER, pady = 5)

instruction4 = Label(text = "Step 4: Select desired checkboxes and click 'Fix'", font=("Helvetica", 18, 'bold'), bg = "black", fg = "white")
instruction4.pack(anchor = CENTER, pady = 5)


app = Window(root)
root.mainloop()

