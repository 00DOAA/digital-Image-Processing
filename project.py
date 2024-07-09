import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np
from scipy.fftpack import fft, fft2, fftshift, ifftshift, ifft2
import cv2
from tkinter import messagebox
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing Tool")

        # Check and print current working directory
        print("Current working directory:", os.getcwd())

        # Set the background image with correct path
        bg_image = Image.open("back.png")  # Update this path if necessary
        bg_image = bg_image.resize((810, 230), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label = tk.Label(master, image=bg_photo)
        self.bg_label.image = bg_photo
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        master.resizable(width=False, height=False)
        master.geometry("810x230")

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_action,
                                       bg="#001F3F", fg="white", width=20, height=4, font=("Helvetica", 12, "bold"))
        self.upload_button.pack(pady=90)

    def upload_action(self):
        file_path = filedialog.askopenfilename(title="Select an image file",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.image_path = file_path
            self.noisy_image = None
            self.perodic_noise = None
            self.create_image_processing_buttons()

    def create_image_processing_buttons(self):
        self.upload_button.destroy()

        # Buttons for image processing actions
        actions = [
            ("Image Histogram", self.calculate_histogram),
            ("Equalize Image Histogram", self.histogram_equalization),
            ("Apply Sobel Filter", self.apply_sobel_filter),
            ("Apply Laplace Filter", self.apply_laplace_filter),
            ("Image Fourier Transform", self.apply_fourier_transform),
            ("Add Salt and Pepper Noise", self.add_salt_and_pepper_noise),
            ("Remove Salt and Pepper Noise", self.remove_salt_and_pepper_noise),
            ("Add Periodic Noise", self.add_periodic_noise),
            ("Remove Periodic Noise", self.remove_periodic_noise)
        ]

        row_num = 0
        col_num = 0

        for action_name, action_func in actions:
            button = tk.Button(self.master, text=action_name, command=action_func, bg="#101314", fg="white", width=25,
                               height=2, font=("Helvetica", 12, "bold"))
            button.grid(row=row_num, column=col_num, padx=5, pady=10)

            col_num += 1
            if col_num > 2:
                col_num = 0
                row_num += 1

    def calculate_histogram(self):
        # Read the image
        image = cv2.imread(self.image_path, 0)

        plt.figure(figsize=(10, 5))

        # Display the image
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('original Image'), plt.xticks([]), plt.yticks([])

        # Calculate and display the histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        plt.subplot(122), plt.plot(hist)
        plt.title("Histogram"), plt.xlabel('Pixel Value'), plt.ylabel('Frequency')

        plt.show()

    def histogram_equalization(self):
        image = cv2.imread(self.image_path, 0)
        equ = cv2.equalizeHist(image)

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow(equ, cmap="gray")
        plt.title("Equalized Image")

        hist, bins = np.histogram(equ.flatten(), 256, [0, 256])
        plt.subplot(122), plt.plot(hist)
        plt.title("Equalized Histogram"), plt.xlabel('Pixel Value'), plt.ylabel('Frequency')

        plt.show()

    def apply_sobel_filter(self):
        ksize = simpledialog.askinteger("Kernel Size", "Enter the Sobel filter kernel size : ")

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
        ksize = simpledialog.askinteger("Kernel Size", "Enter the Laolace filter kernel size:")

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
        plt.subplot(122), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), cmap="gray"), plt.title(
            "Noisy Image")
        plt.show()

    def remove_salt_and_pepper_noise(self):
        noisy_image = self.noisy_image
        if noisy_image is not None:
            kernel_size = simpledialog.askinteger("Kernel Size", "Enter the Median filter kernel size : ")
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
        noise = amplitude * np.sin(2 * np.pi * frequency * y / height)
        noisy_image = image + noise.astype(np.uint8)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        self.periodic_noise_image = noisy_image

        plt.subplot(121), plt.imshow(image, cmap="gray"), plt.title("Original Image")
        plt.subplot(122), plt.imshow(noisy_image, cmap="gray"), plt.title("Noisy Image")
        plt.show()

    def remove_periodic_noise(self):
        if self.periodic_noise_image is None:
            messagebox.showwarning("Warning", "Noisy Image Not Found.")
        else:
            image = self.periodic_noise_image
            f = fft2(image)
            fshift = fftshift(f)
            magnitude_spectrum = np.abs(fshift)

            plt.subplot(121), plt.imshow(image, cmap='gray')
            plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(122), plt.imshow(20 * np.log(magnitude_spectrum), cmap='gray')
            plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

            plt.show()

            mask = np.ones(image.shape, np.uint8)
            peaks, _ = find_peaks(magnitude_spectrum.ravel(), height=50)
            for peak in peaks:
                y, x = np.unravel_index(peak, image.shape)
                mask[y - 10:y + 10, x - 10:x + 10] = 0

            fshift = fshift * mask
            f_ishift = ifftshift(fshift)
            img_back = ifft2(f_ishift)
            img_back = np.abs(img_back)

            plt.subplot(121), plt.imshow(image, cmap='gray')
            plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(122), plt.imshow(img_back, cmap='gray')
            plt.title('Restored Image'), plt.xticks([]), plt.yticks([])

            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
