from PIL import Image

# Open the image file
im = Image.open('image.jpg')

# Convert the image to a matrix of pixel values
pixels = im.load()

# Create a new image with the same size as the original
new_im = Image.new('RGB', im.size)
new_pixels = new_im.load()

# Define the size of the blur kernel
kernel_size = 3

# Iterate over every pixel in the image
for i in range(im.size[0]):
    for j in range(im.size[1]):
        # Initialize a sum for the pixel values
        pixel_sum = (0, 0, 0)

        # Iterate over the pixels in the kernel
        for ii in range(-kernel_size//2, kernel_size//2+1):
            for jj in range(-kernel_size//2, kernel_size//2+1):
                # Check that the kernel does not extend outside the image
                if i + ii >= 0 and i + ii < im.size[0] and j + jj >= 0 and j + jj < im.size[1]:
                    # Add the pixel value to the sum
                    pixel_sum = (pixel_sum[0] + pixels[i+ii, j+jj][0], pixel_sum[1] + pixels[i+ii, j+jj][1], pixel_sum[2] + pixels[i+ii, j+jj][2])
        # Divide the sum by the number of pixels in the kernel to get the average value
        avg_pixel = (pixel_sum[0] // (kernel_size**2), pixel_sum[1] // (kernel_size**2), pixel_sum[2] // (kernel_size**2))
        # Set the pixel value in the new image
        new_pixels[i, j] = avg_pixel

# Save the new image
new_im.save('blurred_image.jpg')