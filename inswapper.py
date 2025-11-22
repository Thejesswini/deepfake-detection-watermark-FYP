import cv2
import torch
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import matplotlib.pyplot as plt

def inswapper_128(path, sources, targets, show_plot=False):
    # --- Setup: This part runs only once ---
    # Initialize the FaceAnalysis app to detect faces
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Initialize the FaceSwapper model
    # This will automatically download the model if you don't have it
    model_path = r"C:\Users\theju\.insightface\models\inswapper_128.onnx"
    swapper = insightface.model_zoo.get_model(model_path)
    # --- End of Setup ---


    def generate_deepfake(source_img, target_img):
        """
        Generates a deepfake by swapping the face from a source image onto a target image.

        Args:
            source_img (np.ndarray): The source image (from OpenCV) with the face to use.
            target_img (np.ndarray): The target image (from OpenCV) to swap the face onto.

        Returns:
            np.ndarray: The resulting deepfaked image as a NumPy array (OpenCV BGR format).
                        Returns the original target image if no faces are found.
        """
        # Detect faces in the source image
        source_faces = app.get(source_img)
        if not source_faces:
            print("Warning: No face found in the source image.")
            return target_img

        # Detect faces in the target image
        target_faces = app.get(target_img)
        if not target_faces:
            print("Warning: No face found in the target image.")
            return target_img

        # Perform the swap using the first detected face from each image
        # The `get` method handles the face swapping and pastes it back.
        res_img = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

        return res_img

    for source, target in zip(sources, targets):
        # --- Example Usage ---
        source_image = cv2.imread(f"{path}{source}.png")
        target_image = cv2.imread(f"{path}{target}.png")

        # Check if images were loaded correctly
        if source_image is None or target_image is None:
            print("Error: Could not load one or both images. Check the file paths.")
        else:
            # Generate the deepfake
            deepfake_result = generate_deepfake(source_image, target_image)
            cv2.imwrite(f".\\inswapper_results\\inswapper_{source}_{target}.png", deepfake_result)

            # Display the result (optional)
            if show_plot:
                # Convert images from BGR (OpenCV) to RGB (Matplotlib) for correct color display
                source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                deepfake_rgb = cv2.cvtColor(deepfake_result, cv2.COLOR_BGR2RGB)

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(source_rgb)
                axs[0].set_title('Source Face')
                axs[0].axis('off')

                axs[1].imshow(target_rgb)
                axs[1].set_title('Target Image')
                axs[1].axis('off')
                
                axs[2].imshow(deepfake_rgb)
                axs[2].set_title('Deepfake Result')
                axs[2].axis('off')

                plt.show()

if __name__=='__main__':
    swaps = [(14, 56), (40, 17), (9, 42), (30, 48), (28, 52), (3, 34), (53, 8), (46, 10), (18, 59), (29, 44)]
    sources = [i for i,j in swaps]
    targets = [j for i,j in swaps]
    
    sources = [29]
    target = [44]
    for source, target in swaps:
        inswapper_128(path=f".\\outputs\\watermarked\\watermarked_", sources=sources, targets=targets)
        break