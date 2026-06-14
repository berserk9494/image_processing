import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import cv2
from skimage import exposure
from skimage.restoration import denoise_tv_chambolle
def disp_img(img , title = 'img' ,text = {'text' : [None],'loc':[(165,500)]}):
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONUP:
            cv2.destroyAllWindows()
    
    I = img.copy()
    avg = np.mean(I)
    for  i , val  in  enumerate(text['text']):
        if avg> 100:
            cv2.putText(I, val, text['loc'][i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:  
            cv2.putText(I, val, text['loc'][i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 

    cv2.imshow(title ,I)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)

    # Associate the callback function with the named window
    cv2.setMouseCallback(title, mouse_callback)


    ########################################### Convert Color Spaces #####################################
def BGRtoHSV(BGR):
    hsv = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
    return cv2.split(hsv)

def HSVtoBGR(H,S,V):
    hsv = np.stack([H,S,V],axis=2)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

######################################## Adaptive Gamma Correction ###################################
def adaptive_gamma_transform(img, n, m):
    """
    Adaptive gamma on TV-denoised V-channel (paper Eq. 15).

    Id(x,y) = I_V_u(x,y) ^ (N(x,y) / I_V_u(x,y) + b(x,y))
    """
    # Previous loop implementation (incorrect gamma: N/pixel + b instead of N/(pixel + b)):
    # rows, cols = img.shape
    # gamma_corrected = np.zeros((rows, cols))
    # img = (img + 1.) / 255.
    # for i in range(rows):
    #     for j in range(cols):
    #         rmin = max(0, i - m // 2)
    #         rmax = min(rows, i + m // 2 + 1)
    #         cmin = max(0, j - n // 2)
    #         cmax = min(cols, j + n // 2 + 1)
    #         local_area = img[rmin:rmax, cmin:cmax]
    #         N = np.mean(local_area)
    #         b = np.var(local_area)
    #         gamma = N / (img[i, j]+ 1e-8) + b
    #         gamma_corrected[i, j] = np.power(img[i, j], gamma)
    # return (gamma_corrected * 255).astype(np.uint8)
    # vectorized implementation
    img_f = (img.astype(np.float64) + 1.) / 255.
    N = cv2.blur(img_f, (n, m), borderType=cv2.BORDER_REFLECT)
    b = cv2.blur(img_f ** 2, (n, m), borderType=cv2.BORDER_REFLECT) - N ** 2 # var[I] = E[I^2] - E[I]^2
    gamma = N / (img_f + 1e-8) + b
    gamma_corrected = np.power(img_f, gamma)
    return np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)

#################################### Guided Filter  ##################################
def _box_filter(img, radius):
    ksize = 2 * int(radius) + 1
    return cv2.boxFilter(img, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, radius, eps):
    """Edge-preserving filter; Id is both guide and src for the log branch."""
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    mean_i = _box_filter(guide, radius)
    mean_p = _box_filter(src, radius)
    mean_ip = _box_filter(guide * src, radius) # E[I*P]
    mean_ii = _box_filter(guide * guide, radius) # E[I^2]
    var_i = mean_ii - mean_i * mean_i # var[I] = E[I^2] - E[I]^2
    cov_ip = mean_ip - mean_i * mean_p # cov[I,P] = E[I*P] - E[I]*E[P]
    a = cov_ip / (var_i + eps) # a = cov[I,P] / (var[I] + eps)
    b = mean_p - a * mean_i # b = E[P] - a*E[I]
    return _box_filter(a, radius) * guide + _box_filter(b, radius) # I_p = a*I + b ,a =cov[y,x]/var[x] , b = E[y] - a*E[x]


def gf_msr(img, radius=15, eps_scales=(0.005, 0.015, 0.025)):
    """Multi-scale log Retinex with guided-filter illumination (paper Sec. 3.B, Il)."""
    img_f = img.astype(np.float64)
    log_stab = 1.0
    il = np.zeros_like(img_f)
    for eps in eps_scales:
        illumination = guided_filter(img_f, img_f, radius, eps)
        il += np.log10(img_f + log_stab) - np.log10(illumination + log_stab)
    il /= len(eps_scales)
    return cv2.normalize(il, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


#################################################### MSR (MSRCR baseline) ##########################
def get_ksize(sigma):
    return max(3, int(((sigma - 0.8) / 0.15) + 2.0))


def get_gaussian_blur(img, ksize=0, sigma=5):
    if ksize == 0:
        ksize = get_ksize(sigma)
    sep_k = cv2.getGaussianKernel(ksize, sigma)
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))


def msr(img, sigma_scales=(15, 80, 250), apply_normalization=True):
    """Classical Gaussian MSR  — used by MSRCR comparison model only."""
    img = img.astype(np.float64) + 1.0
    msr_out = np.zeros_like(img)
    for sigma in sigma_scales:
        blur = get_gaussian_blur(img, ksize=0, sigma=sigma)
        msr_out += np.log10(img) - np.log10(blur + 1.0)
    msr_out /= len(sigma_scales)
    if apply_normalization:
        return cv2.normalize(msr_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return msr_out


def mtanh(img, sigma_scales=(15, 80, 250)):
    """Multi-scale tanh enhancement """
    img_f = img.astype(np.float64) + 1.0
    i_t = np.zeros_like(img_f)
    for sigma in sigma_scales:
        blur = get_gaussian_blur(img_f, ksize=0, sigma=sigma)
        i_t += np.tanh(img_f / (blur + 1e-8))
    i_t /= len(sigma_scales)
    return np.clip(i_t * 255, 0, 255).astype(np.uint8)


################################### Double-Function Image Enhancement ################################
def DFIE(img, sigma=(10, 40, 300), n=3, m=3, gf_radius=15, gf_eps=(0.005, 0.015, 0.025)):
    """Paper Eqs. 18-20: fuse guided-filter log branch Il with tanh branch It."""
    i_l = gf_msr(img, gf_radius, gf_eps).astype(np.float64)
    i_t = mtanh(img, sigma).astype(np.float64)

    i_l_mean = cv2.blur(i_l, (n, m), borderType=cv2.BORDER_REFLECT)
    i_t_mean = cv2.blur(i_t, (n, m), borderType=cv2.BORDER_REFLECT)
    alpha = np.clip(i_t_mean / (i_l_mean + 1e-8), 0, 1)

    balanced = alpha * i_l + (1 - alpha) * i_t
    return np.clip(balanced, 0, 255).astype(np.uint8)


################################## Three-Dimensional Gamma Correction ################################
def three_dim_gamma_correction(image, weights=(0.1, 0.1, 0.05), n=3, m=3):
    """Paper Eq. 22: Iout = Id ^ (psi*exp(A) + mu*exp(B) + nu*exp(C))."""
    psi, mu, nu = weights
    img_f = (image.astype(np.float64) + 1.) / 255.

    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy) # gradient magnitude |sqrt(gx^2 + gy^2)|

    # A: Id(x,y) / max in m x n window (dilate = local max)
    local_max = cv2.dilate(img_f, np.ones((m, n), np.uint8), borderType=cv2.BORDER_REFLECT)
    A = img_f / (local_max + 1e-8)

    # B: local mean gradient 
    B = cv2.blur(grad_mag, (n, m), borderType=cv2.BORDER_REFLECT) # E[|grad(I)|] = 1/(m*n) * sum( |grad(I)| )

    # C: local variance 
    mean_f = cv2.blur(img_f, (n, m), borderType=cv2.BORDER_REFLECT)
    C = cv2.blur(img_f ** 2, (n, m), borderType=cv2.BORDER_REFLECT) - mean_f ** 2 # var[I] = E[I^2] - E[I]^2

    gamma = psi * np.exp(A) + mu * np.exp(B) + nu * np.exp(C)
    return np.clip(np.power(img_f, gamma) * 255, 0, 255).astype(np.uint8)

#################################### Adaptive Saturation  Correction #################################

def adaptive_saturation_adjustment(s_channel, n, m):
    """Paper Eq. 25: multiply denoised S by a local factor (not min-max stretch)."""
    s_f = (s_channel.astype(np.float64) + 1.0) / 255.0
    sx = cv2.Sobel(s_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(s_f, cv2.CV_64F, 0, 1, ksize=3)
    Sg = np.hypot(sx, sy)
    S_mean = np.mean(s_f)
    Sm = cv2.boxFilter(s_f, -1, (n, m), borderType=cv2.BORDER_REFLECT) # same as cv2.blur(s_f, (n, m), borderType=cv2.BORDER_REFLECT)
    dark_mask = s_f <= (S_mean + Sg)
    factor = np.where(
        dark_mask,
        1.0 + 0.8 * np.log10(Sm / (s_f + 0.5 * Sg + 1e-8)),
        np.exp((Sm - s_f) / 2.0),
    )
    return np.clip(s_channel.astype(np.float64) * factor, 0, 255).astype(np.uint8)

############################### Multi Scale Retinex with Color Restoration ###########################
def color_balance(img, low_per, high_per):
    '''Contrast stretch img by histogram equilization with black and white cap'''
    
    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    # channels of image
    ch_list = []
    if len(img.shape) == 2:
        ch_list = [img]
    else:
        ch_list = cv2.split(img)
    
    cs_img = []
    # for each channel, apply contrast-stretch
    for i in range(len(ch_list)):
        ch = ch_list[i]
        # cummulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array([0 if i < li 
                        else (255 if i > hi else round((i - li) / (hi - li) * 255)) 
                        for i in np.arange(0, 256)], dtype = 'uint8')
        # constrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)
    
    if len(cs_img) == 1:
        return np.squeeze(cs_img)
    elif len(cs_img) > 1:
        return cv2.merge(cs_img)
    return None

def msrcr(img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    # Multi-scale retinex with Color Restoration
    # MSRCR(x,y) = G * [MSR(x,y)*CRF(x,y) - b], G=gain and b=offset
    # CRF(x,y) = beta*[log(alpha*I(x,y) - log(I'(x,y))]
    # I'(x,y) = sum(Ic(x,y)), c={0...k-1}, k=no.of channels
    
    img = img + 1.0
    # Multi-scale retinex and don't normalize the output
    msr_img = msr(img, sigma_scales, apply_normalization=False)
    # Color-restoration function
    crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
    # MSRCR
    msrcr = G * (msr_img*crf - b)
    # normalize MSRCR
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # color balance the final MSRCR to flat the histogram distribution with tails on both sides
    msrcr = color_balance(msrcr, low_per, high_per)
    
    return msrcr


################################################# CLAHE ##############################################
def CLAHE(Img):
    # Convert image to LAB color space
    lab = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)

    # Split LAB image into separate channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel with the other LAB channels
    lab_cl = cv2.merge((cl,a,b))

    # Convert back to RGB color space
    final = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)
    return final 

################################### Adaptive histogram equalization ##################################
def AHE(Img):
    eq = exposure.equalize_adapthist(Img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    
################################################ Metrics #############################################
def PSNR(I_r, I_f):
    mse = np.mean((I_r - I_f) ** 2)
    max_pixel = 255
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    # psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return round(psnr, 4)


def SD(I_f):
    # Compute the histogram
    hist, bins = np.histogram(I_f.flatten(), bins=256)
    # Compute the mean of the histogram
    mean = np.sum(hist * bins[:-1]) / np.sum(hist)

    # Compute the variance of the histogram
    variance = np.sum((bins[:-1] - mean) ** 2 * hist) / np.sum(hist)

    # Compute the standard deviation of the histogram
    return round(np.sqrt(variance), 4)



def SSIM(I_r, I_f, L=255):
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    # INITS
    I2_2 = I_f ** 2  # I2^2
    I1_2 = I_r ** 2  # I1^2
    I1_I2 = I_r * I_f  # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I_r, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I_f, (11, 11), 1.5)
    mu1_2 = mu1 ** 2
    mu2_2 = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = t3 / t1
    mssim = np.mean(ssim_map)  # mssim = average of ssim map
    return round(mssim, 4)


def IE(I_f):
    # Add epsilon to avoid division by zero errors
    epsilon = 2**(-32) 
    # Compute the histogram of the image
    hist, _ = np.histogram(I_f.flatten(), bins=256)

    # Calculate the total number of pixels in the image
    num_pixels = np.sum(hist) # same as N*M

    # Calculate the PMF by dividing each bin in the histogram by the total number of pixels
    hist_p = hist / num_pixels

    hist_p = np.clip(hist_p, epsilon, 1)
    E = -np.sum(hist_p * np.log2(hist_p))
    return round(E, 4)


def metric(I_ref ,I_enc):
    I_ref_gray = cv2.cvtColor(I_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    I_enc_gray = cv2.cvtColor(I_enc, cv2.COLOR_BGR2GRAY).astype(np.float64)
    info_ref = {'PSNR': PSNR(I_ref_gray, I_enc_gray),'SSIM': SSIM(I_ref_gray, I_enc_gray, L=255) ,'SD': SD(I_enc_gray) ,'IE':IE(I_enc_gray)}
    return info_ref


def Model(Img ,model = 'DFE' ,disp_selector = [False,False,False,False,False ,False,False ,False, False ,False]
           ,sigma = [10,100,300],weights=[0.1,0.1,0.05], kernel = [9,9],lam = 90):
    
    if model == 'DFE':
         # disp_selector = [Original , Original & I_o,  HSV ,I_d,I_p,I_out,I_img,I_u,S_tag ]
         h,s,v = BGRtoHSV(Img)  # Hue channel , Saturation channel , Value channel
         n , m = kernel
         ######################### V - Channel #########################
         #I_v= denoise_tv(v, weight =1/lam, eps=1e-6, max_num_iter=100)
         I_v = np.clip(denoise_tv_chambolle(v.astype(np.float64), weight=1/lam, eps=1e-6, max_num_iter=100),0, 255).astype(np.uint8)
         I_d = adaptive_gamma_transform(I_v, n, m)
         I_p = DFIE(I_d , sigma,n ,m )
         I_out = three_dim_gamma_correction(I_d,weights,n,m)
         I_img = ((I_out/255.*I_p/255.)*255).astype(np.uint8)

         ######################### S - Channel #########################
         #I_u= denoise_tv(s, weight=1/lam, eps=1e-6, max_num_iter=100)
         I_u = np.clip(denoise_tv_chambolle(s.astype(np.float64), weight=1/lam, eps=1e-6, max_num_iter=100),0, 255).astype(np.uint8)
         S_tag = adaptive_saturation_adjustment(I_u, n, m)
         
         ############################# I_o #############################
         I_o = HSVtoBGR(h,S_tag,I_img)
         I_o_cb = color_balance(I_o,1,1)
         performance = metric(Img,I_o)
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o],[I_o_cb]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement','Enhancement + Color Balance'],'loc':[(280,460),(640+280,460),(640*2+230,460)]})     
         if disp_selector[2]:
            disp_img(np.block([h,s,v]) , title = 'HSV' ,text = {'text' : ['h-channel','s-channel','v-channel'],'loc':[(280,460),(640+280,460),(640*2+280,460)]})  
         if disp_selector[3]:
            disp_img(I_d, title = 'I_d' ,text = {'text' : ['I_d'],'loc':[(280,460)]})
         if disp_selector[4]:
            disp_img(I_p, title = 'I_p' ,text = {'text' : ['I_p'],'loc':[(280,460)]})
         if disp_selector[5]:
            disp_img(I_out, title = 'I_out' ,text = {'text' : ['I_out'],'loc':[(280,460)]})
         if disp_selector[6]:
            disp_img(I_img, title = 'I_img' ,text = {'text' : ['I_img'],'loc':[(280,460)]})
         if disp_selector[7]:
            disp_img(I_u, title = 'I_u' ,text = {'text' : ['I_u'],'loc':[(280,460)]})
         if disp_selector[8]:
            disp_img(I_v, title = 'I_v' ,text = {'text' : ['I_v'],'loc':[(280,460)]})
         if disp_selector[9]:
            disp_img(S_tag, title = 'S_tag' ,text = {'text' : ['S_tag'],'loc':[(280,460)]})
         return  performance
    if model == 'MSRCR':
         I_o = msrcr(Img,sigma_scales=sigma) 
         performance = metric(Img,I_o)
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement'],'loc':[(280,460),(640+280,460)]})

         return  performance    
    if model == 'CLAHE':
         I_o = CLAHE(Img) 
         performance = metric(Img,I_o)
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement'],'loc':[(280,460),(640+280,460)]})  

         return  performance            
    if model == 'AHE':
         I_o = AHE(Img)
         performance = metric(Img,I_o) 
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement'],'loc':[(280,460),(640+280,460)]}) 
 
         return  performance   
             

class ImageProcessorGUI:

    def __init__(self, master):
        self.master = master
        master.title("Image Processor")
        w = 380
        h = 370
        # open window in the center of screen
        screen_width = master.winfo_screenwidth()  # get the screen width
        screen_height = master.winfo_screenheight()  # get the screen height
        x = int((screen_width / 2) - (w / 2))
        y = int((screen_height / 2) - (h / 2))
        master.geometry('{}x{}+{}+{}'.format(w, h, x, y))  # window.geometry('wxh+x+y')
            
        # Select image button
        self.select_image_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_image_button.grid(row=0, column=0, padx=10, pady=10)

        # Display Image button
        self.select_image_button = tk.Button(master, text="Display Selected",state='disabled', command= self.display_selected)
        self.select_image_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Model combobox
        self.model_label = tk.Label(master, text="Select Model:")
        self.model_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.model_combobox = ttk.Combobox(master, values=['MSRCR', 'CLAHE', 'AHE', 'DFE'],textvariable= 'Select',state='disabled')
        self.model_combobox.bind("<<ComboboxSelected>>", self.en)
        self.model_combobox.grid(row=1, column=1, padx=10, pady=5)
        
        # Value entries
        self.values_label_sigma = tk.Label(master, text="Enter Sigma values :")
        self.values_label_sigma.grid(row=2, column=0, padx=10, pady=5)
        
        self.values_entry_sigma = tk.Entry(master,state='disabled')
        self.values_entry_sigma.grid(row=2, column=1, padx=10, pady=5 )

        self.values_label_weight = tk.Label(master, text="Enter Weight values :")
        self.values_label_weight.grid(row=3, column=0, padx=10, pady=5)
        
        self.values_entry_weight = tk.Entry(master,state='disabled')
        self.values_entry_weight.grid(row=3, column=1, padx=10, pady=5)

        
        self.values_label_kernel = tk.Label(master, text="Enter Kernel size (n,m) :")
        self.values_label_kernel.grid(row=4, column=0, padx=10, pady=5)
        
        self.values_entry_kernel= tk.Entry(master,state='disabled')
        self.values_entry_kernel.grid(row=4, column=1, padx=10, pady=5)

        
        self.values_label_lambda = tk.Label(master, text="Enter Lambda value :")
        self.values_label_lambda.grid(row=5, column=0, padx=10, pady=5)
        
        self.values_entry_lambda = tk.Entry(master,state='disabled')
        self.values_entry_lambda.grid(row=5, column=1, padx=10, pady=5)
        
        # Checkboxes
        self.checkbox_frame = tk.Frame(master)
        self.checkbox_frame.grid(row=6, column=1, padx=10)
       
        self.checkbox_labels = ['I_o','I_i&o','HSV', 'I_d', 'I_p', 'I_out', 'I_img', 'I_u','I_v', "S'"]
        self.checkbox_vars = [tk.BooleanVar() for i in range(len(self.checkbox_labels))]
        self.checkbox_buttons = []
        
        for i in range(len(self.checkbox_labels)):
            self.checkbox_buttons.append(tk.Checkbutton(self.checkbox_frame, text=self.checkbox_labels[i], variable=self.checkbox_vars[i],state='disabled'))
            if i < len(self.checkbox_labels) // 2:
                self.checkbox_buttons[i].grid(row=i, column=0, sticky='w')
            else:
                self.checkbox_buttons[i].grid(row=i-len(self.checkbox_labels) // 2, column=1, sticky='w')
        
        self.metric_table = ttk.Frame(master)
        self.metric_table.grid(row=6, column=0, padx=5)
        self.metric_table_label = ttk.Label(self.metric_table, text="Metric Table")
        self.metric_table_label.grid(row=7, column=0, pady=(5, 5))

        self.metric_table_treeview = ttk.Treeview(self.metric_table, height=5)
        self.metric_table_treeview.grid(row=8, column=0)
        self.metric_table_treeview['columns'] = ("Metric-Name", "Metric-Value")

        # format columns
        self.metric_table_treeview.column("#0", width=0, stretch=False)
        self.metric_table_treeview.column("Metric-Name", width=100, minwidth=100, anchor="center")
        self.metric_table_treeview.column("Metric-Value", width=100, minwidth=100, anchor="center")

        # create headings
        self.metric_table_treeview.heading("#0", text="", anchor="w")
        self.metric_table_treeview.heading("Metric-Name", text="Metric-Name", anchor="center")
        self.metric_table_treeview.heading("Metric-Value", text="Metric-Value", anchor="center")
        

        # Run button
        self.run_button = tk.Button(self.checkbox_frame, text="Run",state='disabled', command=self.run)
        self.run_button.grid(row=7, column=0, padx=10, pady=5 )

          
    
    def en(self, event):
        self.run_button.config(state='normal')
        if self.model_combobox.get() == 'MSRCR':
           self.values_entry_sigma.configure(state='normal')
           self.values_entry_sigma.delete(0,'end')
           self.values_entry_sigma.insert(1,'10,100,300')
           self.values_entry_weight.configure(state='disable')
           self.values_entry_kernel.configure(state='disable')
           self.values_entry_lambda.configure(state='disable')
           for i in range(len(self.checkbox_buttons)):
               self.checkbox_buttons[i].configure(state='disable')
               self.checkbox_buttons[i].deselect()
           self.checkbox_buttons[0].configure(state='normal')
           self.checkbox_buttons[1].configure(state='normal')

        elif  self.model_combobox.get() == 'DFE': 
           self.values_entry_sigma.configure(state='normal')
           self.values_entry_sigma.delete(0,'end')
           self.values_entry_sigma.insert(1,'10,100,300')
           self.values_entry_weight.configure(state='normal')
           self.values_entry_weight.delete(0,'end')
           self.values_entry_weight.insert(1,'0.12,0.12,0.22')
           self.values_entry_kernel.configure(state='normal')
           self.values_entry_kernel.delete(0,'end')
           self.values_entry_kernel.insert(1,'9,9')
           self.values_entry_lambda.configure(state='normal')
           self.values_entry_lambda.delete(0,'end')
           self.values_entry_lambda.insert(1,'90')
           for i in range(len(self.checkbox_buttons)):
               self.checkbox_buttons[i].configure(state='normal') 

        else:
           self.values_entry_sigma.configure(state='disable')
           self.values_entry_weight.configure(state='disable')
           self.values_entry_kernel.configure(state='disable')
           self.values_entry_lambda.configure(state='disable')
           for i in range(len(self.checkbox_buttons)):
               self.checkbox_buttons[i].configure(state='disable')
               self.checkbox_buttons[i].deselect()
           self.checkbox_buttons[0].configure(state='normal')
           self.checkbox_buttons[1].configure(state='normal')

        for i in self.metric_table_treeview.get_children():
              self.metric_table_treeview.delete(i)         
        
    def select_image(self):
        file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.tif;*.bmp;")])
        if file_paths:
            image = cv2.imread(file_paths[0])
            self.image = cv2.resize(image, (640, 480))
            self.select_image_button.config(state='normal')
            self.model_combobox.config(state='readonly')
        else:
            self.image = None    

    def display_selected(self):
        disp_img(self.image , title = 'Reference' ,text = {'text' : ['Original'],'loc':[(280,460)]})
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

    def insert_data_to_metric_table(self):
        for i in self.metric_table_treeview.get_children():
            self.metric_table_treeview.delete(i)

        for idx, (key, value) in enumerate(self.info.items()):
            self.metric_table_treeview.insert(parent='', index='end', iid=str(idx), values=(key, value))      
    
    def run(self):
        # Get values from GUI elements
        sigma =  None
        weights = None
        kernel = None 
        lam = None
        Img = self.image
        model = self.model_combobox.get()
 
        if model == 'DFE':
           sigma = [int(num)for num in self.values_entry_sigma.get().split(',')]
           weights = [float(num)for num in self.values_entry_weight.get().split(',')]
           kernel = [int(num)for num in self.values_entry_kernel.get().split(',')]
           lam = float(self.values_entry_lambda.get())
        elif model == 'MSRCR': 
           sigma = [int(num)for num in self.values_entry_sigma.get().split(',')] 
        
        
        checkboxes = [var.get() for var in self.checkbox_vars]

        # Display processed image
        self.info = Model(Img ,model =model ,disp_selector = checkboxes,sigma = sigma,weights=weights ,kernel =kernel ,lam = lam)
        self.insert_data_to_metric_table()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
if __name__ == '__main__':
    root = tk.Tk()
    
    gui = ImageProcessorGUI(root)
    root.mainloop()