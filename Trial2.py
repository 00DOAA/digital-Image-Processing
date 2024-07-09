import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import self as self
from PIL import Image, ImageTk
import numpy as np
from pyexpat import features
from scipy.fftpack import fft, fft2, fftshift, ifftshift, ifft2
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing Tool")

        # Set the background image
        bg_image = Image.open("back.png")
        bg_image = bg_image.resize((810, 230), Image.ANTIALIAS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label = tk.Label(master, image=bg_photo)
        self.bg_label.image = bg_photo
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        master.resizable(width=False, height=False)
        master.geometry("810x230")

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_action, bg="#001F3F",
                                       fg="white", width=20, height=4, font=("Helvetica", 12, "bold"))
        self.upload_button.pack(pady=90)

        # Create a list of features or missions as buttons
        self.create_feature_buttons()

    def create_feature_buttons(self):
        features = [
        ("Calculate Histogram", self.calculate_histogram),
        ("Equalize Image Histogram", self.histogram_equalization),
        ("Apply Sobel Filter", self.apply_sobel_filter),
        ("Apply Laplace Filter", self.apply_laplace_filter),
        ("Image Fourier Transform", self.apply_fourier_transform),
        ("Add Salt and Pepper Noise", self.add_salt_and_pepper_noise),
        ("Remove Salt and Pepper Noise", self.remove_salt_and_pepper_noise),
        ("Add Periodic Noise", self.add_periodic_noise),
        ("Remove Periodic Noise", self.remove_periodic_noise)
        ]
        for feature_name, feature_func in features:
           button = tk.Button(self.master, text=feature_name,
                           command=lambda func=feature_func: self.select_image_and_execute(func), bg="#101314",
                           fg="white", width=25, height=2, font=("Helvetica", 12, "bold"))
           button.pack(pady=10)

    def select_image_and_execute(self, feature_func):
       file_path = filedialog.askopenfilename(title="Select an image file",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
       if file_path:
           feature_func(file_path)

    def calculate_histogram(self, image_path):
        image = cv2.imread(image_path, 0)
        plt.figure(figsize=(10, 5))
    # Display the image
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('original Image'), plt.xticks([]), plt.yticks([])
    # Calculate and display the histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        plt.subplot(122), plt.plot(hist)
        plt.title('Histogram'), plt.xlabel('Pixel Value'), plt.ylabel('Frequency')
        plt.show()

    def apply_sobel_filter(self):
        ksize = simpledialog.askinteger("Kernel Size", "Enter the Sobel filter kernel size:")
        if ksize is not None:
            image = cv2.imread(self.image_path, 0)
        # Apply Sobel filter in the X and Y directions
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        # Create a figure with multiple subplots
            plt.figure(figsize=(10, 5))

        # Display the Sobel X result
            plt.subplot(1, 2, 1)
            plt.imshow(sobelx, cmap="gray")
            plt.title(f"Sobel X (Kernel Size: {ksize})")

        # Display the Sobel Y result
            plt.subplot(1, 2, 2)
            plt.imshow(sobely, cmap="gray")
            plt.title(f"Sobel Y (Kernel Size: {ksize})")

        # Show the plots
            plt.show()

    def apply_laplace_filter(self):
       ksize = simpledialog.askinteger("Kernel Size", "Enter the Laplace filter kernel size:")

       image = cv2.imread(self.image_path, 0)
       laplacian = cv2.Laplacian(image, cv2.CV_8U, ksize=ksize)
       plt.imshow(laplacian, cmap="gray")
       plt.title(f"Laplace Filter(Kernel Size: {ksize})")
       plt.show()


    def apply_fourier_transform(self):
       image = cv2.imread(self.image_path, 0)

       spectrum = fft2(image)
       spectrum = fftshift(spectrum)
       freq_Image = 20 * np.log(np.abs(spectrum))
       plt.figure(figsize=(10, 5))
       plt.subplot(1, 2, 1)
       plt.imshow(image, cmap="gray")
       plt.title("Image")

       plt.subplot(1, 2, 2)
       plt.imshow(freq_Image, cmap="gray")
       plt.title("Image Fourier Transform")

       plt.show()

    def add_salt_and_pepper_noise(self):
        noise_ratio = simpledialog.askstring("Noise Ratio", "Enter the Noise Ratio:")
        if noise_ratio is not None:
            try:
                noise_ratio = int(noise_ratio)
            except ValueError:
                try:
                    noise_ratio = float(noise_ratio)
                except ValueError:
                    noise_ratio = None

    # Read the image
        image = cv2.imread(self.image_path)
        noisy_image = image.copy()
        h, w, c = image.shape
        noisy_pixels = int(h * w * noise_ratio)
        for _ in range(noisy_pixels):
            row, col = np.random.randint(0, h), np.random.randint(0, w)
            if np.random.rand() < 0.5:
                noisy_image[row, col] = [0, 0, 0]
            else:
                noisy_image[row, col] = [255, 255, 255]

        self.noisy_image = noisy_image
    # Display the original and noisy images
        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap="gray"), plt.title("Original Image")
        plt.subplot(122), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), cmap="gray"), plt.title("Noisy Image")
        plt.show()

    def remove_salt_and_pepper_noise(self):
        noisy_image = self.noisy_image
        if noisy_image is not None:
            kernel_size = simpledialog.askinteger("Kernel Size", "Enter the Median filter kernel size:")
            denoised_image = cv2.medianBlur(noisy_image, kernel_size)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), cmap="gray")
            plt.title("Noisy Image")

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY), cmap="gray")
            plt.title("Filtered Image")

            plt.show()
        else:
            messagebox.showwarning("Warning", "Noisy Image Not Found.")

    def add_periodic_noise(self):
        image = cv2.imread(self.image_path, 0)
        amplitude = simpledialog.askstring("Amplitude", "Enter the Amplitude of Noise:")
        frequency = simpledialog.askstring("Frequency", "Enter the Frequency of Noise:")

        if amplitude and frequency is not None:
            try:
                amplitude = int(amplitude)
                frequency = int(frequency)
            except ValueError:
                try:
                    amplitude = float(amplitude)
                    frequency = float(frequency)
                except ValueError:
                    pass

        height, width = image.shape[:2]
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        noise = amplitude * np.sin(2 * np.pi * frequency * x / width + 2 * np.pi * frequency * y / height)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        self.perodic_noise = noisy_image

        plt.subplot(121), plt.imshow(image, cmap="gray"), plt.title("Original Image")
        plt.subplot(122), plt.imshow(noisy_image, cmap="gray"), plt.title("Noisy Image")
        plt.show()

    def remove_periodic_noise(self):
        if self.perodic_noise is not None:
            remove_periodic_noise_window = tk.Toplevel(self.master)
            remove_periodic_noise_window.title("Remove Periodic Noise")
            remove_periodic_noise_window.configure(bg="#36454f")
            remove_periodic_noise_window.resizable(width=False, height=False)

            mask_button = tk.Button(remove_periodic_noise_window, text="Mask", command=self.mask, bg="#101314", fg="white",
                                width=20, height=2, font=("Helvetica", 12, "bold"))
            ban_button = tk.Button(remove_periodic_noise_window, text="Band_reject", command=self.band_reject, bg="#101314",
                               fg="white", width=20, height=2, font=("Helvetica", 12, "bold"))

            mask_button.grid(row=0, column=0, padx=10, pady=10)
            ban_button.grid(row=0, column=1, padx=10, pady=10)
        else:
            messagebox.showwarning("Warning", "Noisy Image Not Found.")
    def mask(self):
        f_transform = np.fft.fft2(self.perodic_noise)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

        fig, ax = plt.subplots()
        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title('Fourier Transform Magnitude Spectrum')
        plt.show(block=False)

        coordinates = []

        def select_pixels(event):
            if event.inaxes == ax and event.button == 1:
                x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                print(f'Selected pixel at ({x}, {y})')
                coordinates.append((x, y))

                if len(coordinates) == 2:
                    plt.close()

        fig.canvas.mpl_connect('button_press_event', select_pixels)
        plt.show()
    # Apply a Mask based on selected pixels
        (x1, y1), (x2, y2) = coordinates
        mask = np.ones_like(f_transform_shifted, dtype=np.uint8)
        mask[int(y1) - 5:int(y1) + 5, int(x1) - 5:int(x1) + 5] = 0
        mask[int(y2) - 5:int(y2) + 5, int(x2) - 5:int(x2) + 5] = 0

        f_transform_filtered = f_transform_shifted * mask
        f_transform_inverse = np.fft.ifftshift(f_transform_filtered)
        noisy_image_filtered = np.fft.ifft2(f_transform_inverse).real

        plt.figure()
        plt.subplot(121), plt.imshow(self.perodic_noise, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(noisy_image_filtered, cmap='gray')
        plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

        plt.show()

    def band_reject(self):
        center_freq1 = 0.1
        center_freq2 = 0.2
        notch_width = 0.02

        image = cv2.imread(self.image_path, 0)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), dtype=np.uint8)
        radius1 = int(notch_width * rows)
        radius2 = int(notch_width * cols)
        center1 = (int(center_freq1 * rows), int(center_freq1 * cols))
        center2 = (int(center_freq2 * rows), int(center_freq2 * cols))
        cv2.circle(mask, center1, radius1, 0, -1)
        cv2.circle(mask, center2, radius2, 0, -1)

        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        f_transform_filtered = f_transform_shifted * mask

        f_ishift = np.fft.ifftshift(f_transform_filtered)
        filtered_image = np.fft.ifft2(f_ishift).real

        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.perodic_noise, 'gray')
        plt.title("noisy image")

        fig.add_subplot(1, 2, 2)
        plt.imshow(filtered_image, "gray")
        plt.title("image after Notch band reject")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
