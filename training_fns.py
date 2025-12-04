'''
import torch
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
from config import W_EXTRACTION, W_IMPERCEPTIBILITY, BATCH_SIZE

# NOTE: CHANGE THE WAY LOSS IS PRINTED DURING TRAINING!!!

def training_fn_standard(EPOCHS, model, train_loader, device, criterion1, criterion2, optimizer):
    """
    This function takes the input batches,
    Concatenates the watermark,
    Trains forward pass,
    Uses transformed watermark with .detach() for inverse pass
    Uses criterion1 for imperceptibility and criterion2 for watermark robustness loss
    
    NOTE: the negative of criterion1 will be minimized
    """
    data_iter = iter(train_loader)   # Create an iterator
    first_batch = next(data_iter)
    watermarks = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2)).to(device) #assuming that the length and width are same
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            #print("images shape", images.shape)
            
            #print("watermark shape:", watermarks.shape)
            
            input_tensor = torch.cat([images, watermarks], dim=1)
            #print("ip tensor shape:", input_tensor.shape)

            
            embedded = model(input_tensor)
            embedded_image, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)
            loss_imperceptibility = criterion1(embedded_image, images)
            
            corrupted_image = apply_corruptions(embedded_image)
            
            # detach the watermark part to prevent gradients from flowing back in a confusing loop
            corrupted_input_for_inverse = torch.cat([corrupted_image, embedded_watermark_part.detach()], dim=1)
            
            recovered = model.inverse(corrupted_input_for_inverse)
            _, recovered_watermark = torch.chunk(recovered, 2, dim=1)
            loss_extraction = criterion2(recovered_watermark, watermarks)
            
            # TOTAL LOSS
            loss = -(W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Total Loss: {loss.item():.6f}")
            if batch_idx % 100 == 0:
                print(f"Epoch:{epoch} | Imperceptibility: {loss_imperceptibility.item():.6f}, Extraction: {loss_extraction.item():.6f}")
    torch.save(model.state_dict(), "model_weights.pth")

def training_fn_set_trnfrmd_wtmk_2_0(EPOCHS, model, train_loader, device, criterion1, criterion2, optimizer):
    """
    This function takes the input batches,
    Concatenates the watermark,
    Trains forward pass,
    Uses 0 matrix with .detach() instead of transformed watermark for inverse pass
    Uses criterion1 for imperceptibility and criterion2 for watermark robustness loss
    
    NOTE: the negative of criterion1 will be minimized
    """
    data_iter = iter(train_loader)   # Create an iterator
    first_batch = next(data_iter)
    watermarks = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2),first_batch.size(3)).to(device) #assuming that the length and width are same
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            #print("images shape", images.shape)
            
            
            #print("watermark shape:", watermarks.shape)
            
            input_tensor = torch.cat([images, watermarks], dim=1)
            #print("ip tensor shape:", input_tensor.shape)

            
            embedded = model(input_tensor)
            embedded_image, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)
            loss_imperceptibility = criterion1(embedded_image, images)
            
            corrupted_image = apply_corruptions(embedded_image)
            
            # detach the watermark part to prevent gradients from flowing back in a confusing loop
            corrupted_input_for_inverse = torch.cat([corrupted_image, torch.zeros_like(embedded_watermark_part).detach()], dim=1)
            
            recovered = model.inverse(corrupted_input_for_inverse)
            _, recovered_watermark = torch.chunk(recovered, 2, dim=1)
            loss_extraction = criterion2(recovered_watermark, watermarks)
            
            # TOTAL LOSS
            loss = -(W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Total Loss: {loss.item():.6f}")
            if batch_idx % 100 == 0:
                print(f"Epoch:{epoch} | Imperceptibility: {loss_imperceptibility.item():.6f}, Extraction: {loss_extraction.item():.6f}")
    torch.save(model.state_dict(), "model_weights.pth")
'''


#gemini -1 st
'''
import torch
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
from config import W_EXTRACTION, W_IMPERCEPTIBILITY, BATCH_SIZE

# NOTE: CHANGE THE WAY LOSS IS PRINTED DURING TRAINING!!!

def training_fn_standard(EPOCHS, model, train_loader, device, criterion1, criterion2, optimizer):
    """
    NOTE: This function is KEPT UNMODIFIED as it uses the old training objective.
    """
    data_iter = iter(train_loader)   # Create an iterator
    first_batch = next(data_iter)
    # This call needs to change to get the W_map only for consistency if used:
    # watermarks, _ = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2)).to(device)
    # BUT, since this function is deprecated for the new objective, we leave it as is.
    watermarks = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2), first_batch.size(3)).to(device) # assuming that the length and width are same
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            # ... (rest of standard training_fn_standard remains unchanged) ...

            input_tensor = torch.cat([images, watermarks], dim=1)
            embedded = model(input_tensor)
            embedded_image, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)
            loss_imperceptibility = criterion1(embedded_image, images)
            
            corrupted_image = apply_corruptions(embedded_image)
            
            corrupted_input_for_inverse = torch.cat([corrupted_image, embedded_watermark_part.detach()], dim=1)
            
            recovered = model.inverse(corrupted_input_for_inverse)
            _, recovered_watermark = torch.chunk(recovered, 2, dim=1)
            loss_extraction = criterion2(recovered_watermark, watermarks)
            
            # TOTAL LOSS
            loss = -(W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch:{epoch} | Imperceptibility: {loss_imperceptibility.item():.6f}, Extraction: {loss_extraction.item():.6f}")
    torch.save(model.state_dict(), "model_weights_1.pth")


def training_fn_set_trnfrmd_wtmk_2_0(EPOCHS, model, train_loader, device, criterion1, criterion2, optimizer):
    """
    MODIFIED: Now uses the message_tensor for loss calculation and handles
    the two outputs from generate_watermark_matrix and model.inverse.
    """
    # NOTE: Assuming generate_watermark_matrix now returns (W_map, message_tensor)
    
    # We must call generate_watermark_matrix INSIDE the loop if the watermark changes per batch,
    # but based on the code's original placement, we'll keep it outside for efficiency, 
    # and assume the watermark is constant per epoch or run.
    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    
    # --- CRITICAL CHANGE 1: GET BOTH W_map AND the message_tensor ---
    # We pass a placeholder message_length=256, assuming the model uses it.
    watermarks, message_tensor = generate_watermark_matrix(
        BATCH_SIZE, first_batch.size(2), first_batch.size(3), message_length=256 
    )
    watermarks, message_tensor = watermarks.to(device), message_tensor.to(device)
    # -----------------------------------------------------------------

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            
            # 1. Forward Pass (Embedding uses W_map)
            input_tensor = torch.cat([images, watermarks], dim=1)
            embedded = model(input_tensor)
            embedded_image, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)
            loss_imperceptibility = criterion1(embedded_image, images)
            
            # 2. Corruption
            corrupted_image = apply_corruptions(embedded_image)
            
            # 3. Inverse Pass (Extraction)
            embedded_watermark_part_key = torch.zeros_like(embedded_watermark_part).detach()
            corrupted_input_for_inverse = torch.cat([corrupted_image, embedded_watermark_part_key], dim=1)
            
            # --- CRITICAL CHANGE 2: model.inverse returns the features and the message ---
            _, recovered_message = model.inverse(corrupted_input_for_inverse)
            # The recovered_message is the [B, 256] tensor from the Extractor Head
            
            # 4. Loss Calculation (CRITICAL CHANGE 3: Loss vs. MESSAGE_TENSOR)
            # The recovered_message [B, 256] is compared against the message_tensor [B, 256]
            loss_extraction = criterion2(recovered_message, message_tensor)
            
            # TOTAL LOSS
            loss = -(W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch:{epoch} | Imperceptibility: {loss_imperceptibility.item():.6f}, Extraction: {loss_extraction.item():.6f}")
    torch.save(model.state_dict(), "model_weights_1.pth")
    
'''

# training_fns.py
import torch
import torch.nn as nn
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
from config import W_EXTRACTION, W_IMPERCEPTIBILITY  # make sure these exist

def training_fn_set_trnfrmd_wtmk_2_0(EPOCHS, model, train_loader, device, criterion1, criterion2, optimizer, message_length=256):
    """
    Training function for the transformed watermark pipeline.
    Handles per-batch watermark/message generation, image imperceptibility, 
    and message extraction with normalized BCE loss.
    """
    model.train()
    mse_img = nn.MSELoss().to(device)  # imperceptibility loss

    for epoch in range(EPOCHS):
        epoch_loss_img = 0.0
        epoch_loss_msg = 0.0
        epoch_batches = 0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.size(0)
            H, W = images.size(2), images.size(3)

            # -----------------------------
            # Generate watermark & message
            # -----------------------------
            W_map, message_tensor = generate_watermark_matrix(
                batch_size=batch_size,
                height=H,
                width=W,
                message_length=message_length
            )
            W_map = W_map.to(device)
            message_tensor = message_tensor.to(device)

            # -----------------------------
            # Forward pass (embedding)
            # -----------------------------
            input_tensor = torch.cat([images, W_map], dim=1)  # [B, 6, H, W]
            embedded = model(input_tensor)
            embedded_image, embedded_wm_part = torch.chunk(embedded, 2, dim=1)

            # -----------------------------
            # Imperceptibility loss
            # -----------------------------
            loss_img = mse_img(embedded_image, images)

            # -----------------------------
            # Corruption + Inverse (extraction)
            # -----------------------------
            corrupted_image = apply_corruptions(embedded_image)
            embedded_wm_part_key = torch.zeros_like(embedded_wm_part).detach()
            corrupted_inverse_input = torch.cat([corrupted_image, embedded_wm_part_key], dim=1)

            _, recovered_message_logits = model.inverse(corrupted_inverse_input)

            # -----------------------------
            # Message loss (BCEWithLogitsLoss)
            # -----------------------------
            loss_msg = criterion2(recovered_message_logits, message_tensor) 

            # -----------------------------
            # Total loss with normalized scaling
            # -----------------------------
            total_loss = (W_IMPERCEPTIBILITY * loss_img) + (W_EXTRACTION * loss_msg)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # -----------------------------
            # Logging
            # -----------------------------
            epoch_loss_img += loss_img.item()
            epoch_loss_msg += loss_msg.item()
            epoch_batches += 1

            if batch_idx % 100 == 0:
                try:
                    psnr_val = criterion1(embedded_image, images).item()
                except Exception:
                    psnr_val = None

                if psnr_val is not None:
                    print(f"Epoch:{epoch} Batch:{batch_idx} | PSNR(metric): {psnr_val:.4f} | ImgMSE: {loss_img.item():.6f} | MsgLoss: {loss_msg.item():.6f}")
                else:
                    print(f"Epoch:{epoch} Batch:{batch_idx} | ImgMSE: {loss_img.item():.6f} | MsgLoss: {loss_msg.item():.6f}")

        # End epoch averages
        avg_img = epoch_loss_img / max(1, epoch_batches)
        avg_msg = epoch_loss_msg / max(1, epoch_batches)
        print(f"Epoch {epoch} Summary -> Avg Img MSE: {avg_img:.6f} | Avg Msg Loss: {avg_msg:.6f}")

    # Save final weights
    torch.save(model.state_dict(), "model_weights_1.pth")
    print("✅ Training complete and model weights saved!")
