import os
from PIL import Image
import random
import io
import numpy as np
import cv2
from PIL import Image, ImageFilter
from torchvision import transforms as T

class GaussianNoise:
    def __init__(self, mean=0., std=20.):
        self.mean = mean
        self.std  = std
    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(self.mean, self.std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class SpeckleNoise:
    def __init__(self, std=0.2):
        self.std = std
    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * self.std
        arr = np.clip(arr + arr * noise, 0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

class SaltPepperNoise:
    def __init__(self, amount=0.04, s_vs_p=0.5):
        self.amount = amount
        self.s_vs_p = s_vs_p
    def __call__(self, img):
        arr = np.array(img).copy()
        # salt
        n_salt = np.ceil(self.amount * arr.size * self.s_vs_p)
        coords = tuple(np.random.randint(0, d, int(n_salt)) for d in arr.shape)
        arr[coords] = 255
        # pepper
        n_pep = np.ceil(self.amount * arr.size * (1 - self.s_vs_p))
        coords = tuple(np.random.randint(0, d, int(n_pep)) for d in arr.shape)
        arr[coords] = 0
        return Image.fromarray(arr.astype(np.uint8))
    
class ColorfulSparkle:
    """Sprinkle bright, random‐colored specks."""
    def __init__(self, amount=0.02):
        # fraction of pixels to turn into random colors
        self.amount = amount
    def __call__(self, img):
        arr = np.array(img).copy()
        h, w, c = arr.shape
        n = int(self.amount * h * w)
        xs = np.random.randint(0, w, n)
        ys = np.random.randint(0, h, n)
        colors = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
        arr[ys, xs] = colors
        return Image.fromarray(arr)

class RandomMotionBlur:
    """Apply 1D motion blur with random length & angle."""
    def __init__(self, degree_range=(15, 45), angle_range=(-5, 5)):
        self.min_d, self.max_d = degree_range
        self.min_a, self.max_a = angle_range

    def __call__(self, img):
        arr = np.array(img)
        degree = random.randint(self.min_d, self.max_d)
        angle  = random.uniform(self.min_a, self.max_a)
        # create linear kernel
        kernel = np.zeros((degree, degree), dtype=np.float32)
        kernel[degree // 2, :] = 1.0
        kernel = cv2.warpAffine(
            kernel,
            cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1.0),
            (degree, degree)
        )
        kernel = kernel / kernel.sum()
        blurred = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(blurred)

def downsample_then_up(img, scale=4):
    """Crush resolution by scale× then nearest‐neighbor upsample."""
    w, h = img.size
    small = img.resize((w//scale, h//scale), resample=Image.BILINEAR)
    return small.resize((w, h), resample=Image.NEAREST)

def random_jpeg_compression(img, min_q=5, max_q=25):
    buf = io.BytesIO()
    q   = random.randint(min_q, max_q)
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf)

class SpeckleNoise:
    """Film‐grain style noise proportional to pixel value."""
    def __init__(self, std=0.25):
        self.std = std
    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * self.std
        arr = np.clip(arr + arr * noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


# ——— “Super” Augmentation Pipeline ———

super_augment = T.Compose([
    T.RandomApply([
        T.RandomPerspective(distortion_scale=0.3)
    ], p=0.6),

    # A) Colorful sparkles (bright RGB flecks)
    T.RandomApply([ColorfulSparkle(amount=0.9)], p=0.4),

    # B) Motion‐streak distortion
    T.RandomApply([RandomMotionBlur(degree_range=(20,60),
                                    angle_range=(-10,10))],
                  p=0.7),

    # C) Heavy Gaussian blur
    T.RandomApply([T.GaussianBlur(kernel_size=(35,35),
                                  sigma=(10.0,20.0))],
                  p=0.6),

    # D) Film‐grain speckle
    T.RandomApply([SpeckleNoise(std=0.3)], p=0.5),

    # E) Random Gaussian noise
    T.RandomApply([T.Lambda(lambda img: Image.fromarray(
        np.clip(
            np.array(img).astype(np.float32) +
            np.random.normal(0, 30, img.size[::-1] + (3,)),
            0,255
        ).astype(np.uint8)
    ))], p=0.6),

    # F) Down‐and‐up sampling for blocky quality drop
    T.RandomApply([T.Lambda(lambda img: downsample_then_up(img, scale=4))],
                  p=0.5),

    # G) Salt‐and‐pepper specks
    T.RandomApply([SaltPepperNoise(amount=0.05)], p=0.3),

    # H) JPEG compression artifacts
    T.RandomApply([T.Lambda(lambda img: random_jpeg_compression(img,
                                                                min_q=5,
                                                                max_q=20))],
                  p=0.6),

])

def process_images():
    # Define source and destination directories
    source_dir = 'augmented_license_plates_10k'
    dest_dir = 'super_augmented_license_plates_10k'
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Get all image files from source directory

    image_files = [f for f in os.listdir(source_dir)]
    
    # Process each image
    count = 0 
    for img_file in image_files:
        try:
            # Open image
            img_path = os.path.join(source_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            
            # Apply super augmentation
            augmented_img = super_augment(img)
            
            # Save augmented image
            dest_path = os.path.join(dest_dir, img_file)
            augmented_img.save(dest_path)
            
            count +=1 
            if count % 100 == 0:
                print(f"Copied {count} images so far...")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

if __name__ == "__main__":
    process_images() 