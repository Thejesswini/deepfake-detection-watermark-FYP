from revnet_model import RevNet3
import torch
from watermark_generation import generate_watermark_matrix
from PIL import Image
from torchvision import transforms

def embed_watermark_on_images(model_path, image_paths):
    """
    Loads a trained RevNet model and embeds a watermark into a list of images.

    Args:
        model_path (str): Path to the saved model's .pth state_dict file.
        image_paths (list of str): A list of file paths for the images.

    Returns:
        list of PIL.Image: A list of watermarked images in PIL format.
    """
    # 1. Setup: Load model and define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate the model architecture
    model = RevNet3(channels=6).to(device)
    
    # Load the trained weights
    # map_location ensures it works whether you have a GPU or not
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move model to the correct device and set to evaluation mode
    model.to(device)
    model.eval()

    # 2. Define image preprocessing
    # This converts images to tensors and normalizes them to [0, 1]
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    # 3. Process the images
    watermarked_images_pil = []
    to_pil = transforms.ToPILImage()

    with torch.no_grad(): # Disable gradients for inference
        for image_path in image_paths:
            print(f"Processing {image_path}...")
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device) # Add batch dimension

            # Generate the watermark
            watermarks = generate_watermark_matrix(1, image_tensor.size(2), image_tensor.size(3)).to(device)
            
            # Create the input for the model
            input_tensor = torch.cat([image_tensor, watermarks], dim=1)
            
            # --- Perform the forward pass to embed the watermark ---
            embedded_output = model(input_tensor)
            
            # Extract just the image part of the output
            embedded_image_tensor, _ = torch.chunk(embedded_output, 2, dim=1)
            
            # --- Convert the output tensor back to a PIL image ---
            # Remove the batch dimension and move to CPU
            output_tensor = embedded_image_tensor.squeeze(0).cpu()
            output_pil = to_pil(torch.clamp(output_tensor, 0, 1)) # Clamp to ensure valid range
            
            watermarked_images_pil.append(output_pil)
    
    print("Watermarking complete.")
    return watermarked_images_pil

# --- Example of how to use the function ---
if __name__ == '__main__':

    # List of images you want to watermark
    path = r".\celebA\img_align_celeba\img_align_celeba\\"
    paths = []
    for i in range(5):
        paths+= [path+"00000"+str(i+1)+".jpg"]
    
    # Call the function
    watermarked_results = embed_watermark_on_images(r"d:\SSN\DEEPFAKE\models\peano_1000_0.8_0.1\model_weights.pth", paths)
    
    # Save or display the results
    for i, img in enumerate(watermarked_results):
        img.save(f"watermarked_{paths[i].split('\\')[-1]}.png")
        # img.show()


