import torch
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
from config import W_EXTRACTION, W_IMPERCEPTIBILITY, BATCH_SIZE
from evaluate import run_phase1_sanity_check, plot_for_one_img

# NOTE: CHANGE THE WAY LOSS IS PRINTED DURING TRAINING!!!
TEST_IMG_PATH = r"D:\SSN\DEEPFAKE\code\celebA\img_align_celeba\img_align_celeba\000002.jpg"

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
        
    torch.save(model.state_dict(), "model_weights_n.pth")

def training_fn_set_trnfrmd_wtmk_2_0_sanity_check(EPOCHS, model, train_loader, device, criterion1, criterion2, optimizer):
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

        if epoch % 10 == 0:
            passed = run_phase1_sanity_check(model, TEST_IMG_PATH, device, epoch)
            
            if passed:
                # Save the "Sanity Proven" model
                torch.save(model.state_dict(), "phase1_sanity_passed.pth")
                break # Exit the loop

def training_fn_with_both_criterion_as_mse_and_channel_mixing(EPOCHS, model, train_loader, device, optimizer, criterion):
    # Use MSE for both checks. It is much more stable.
    mse_criterion = criterion
    
    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    # Generate watermark once (fixed for the batch setup)
    fixed_watermark = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2), first_batch.size(3)).to(device)
    
    for epoch in range(EPOCHS):
        model.train()
        running_psnr = 0.0
        running_ext_mse = 0.0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            current_batch_size = images.size(0)
            
            # Adjust watermark batch size if last batch is smaller
            if current_batch_size != BATCH_SIZE:
                watermarks = fixed_watermark[:current_batch_size]
            else:
                watermarks = fixed_watermark
            
            # 1. Forward Pass (Embed)
            input_tensor = torch.cat([images, watermarks], dim=1)
            embedded_full = model(input_tensor)
            
            # The model output is mixed, but we want the first 3 channels to look like the image
            # Note: Because of Invertible1x1Conv, 'embedded_image' is not strictly separated anymore
            # But we force the first 3 channels to resemble the image via Loss.
            embedded_image = embedded_full[:, :3, :, :]
            
            # Loss 1: Imperceptibility (Minimize MSE between Input and Embedded)
            loss_imperceptibility = mse_criterion(embedded_image, images)
            
            # 2. Attack Simulation
            corrupted_image = apply_corruptions(embedded_image)
            
            # 3. Inverse Pass (Extract)
            # Create the "Attack" tensor: [Corrupted Image, Zeros]
            # This represents losing the watermark data and trying to recover it from the image traces
            attack_tensor = torch.cat([corrupted_image, torch.zeros_like(watermarks)], dim=1)
            
            recovered_full = model.inverse(attack_tensor)
            recovered_watermark = recovered_full[:, 3:, :, :] # The last 3 channels should be the watermark
            
            # Loss 2: Extraction (Minimize MSE between Original Watermark and Recovered)
            loss_extraction = mse_criterion(recovered_watermark, watermarks)
            
            # Total Loss
            loss = (W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion (common in RevNets)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # --- Logging Metrics ---
            # Calculate PSNR for display only
            with torch.no_grad():
                psnr = 10 * torch.log10(1 / loss_imperceptibility)
                running_psnr += psnr.item()
                running_ext_mse += loss_extraction.item()

            if batch_idx % 100 == 0:
                print(f"Epoch:{epoch} | Loss: {loss.item():.4f} | PSNR: {psnr:.2f}dB | Ext_MSE: {loss_extraction.item():.4f}")

        # End of Epoch Sanity Check
        if epoch % 5 == 0:
            # You can call your visualization function here
            print(f"--- Epoch {epoch} Avg PSNR: {running_psnr/len(train_loader):.2f} ---")
            passed = run_phase1_sanity_check(model, TEST_IMG_PATH, device, epoch)
            
            if passed:
                # Save the "Sanity Proven" model
                torch.save(model.state_dict(), "channel_mixing_phase1_sanity_passed.pth")
                break # Exit the loop

import torch.optim as optim

def training_fn_refined(EPOCHS, model, train_loader, device, criterion, optimizer):
    # 1. Use MSE for both.
    mse_criterion = criterion     
    
    # 3. Setup Optimizer and Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    fixed_watermark = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2), first_batch.size(3)).to(device)
    
    print("--- Starting Extended Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            current_batch_size = images.size(0)
            
            if current_batch_size != BATCH_SIZE:
                watermarks = fixed_watermark[:current_batch_size]
            else:
                watermarks = fixed_watermark
            
            # --- Forward ---
            input_tensor = torch.cat([images, watermarks], dim=1)
            embedded_full = model(input_tensor)
            embedded_image = embedded_full[:, :3, :, :]
            
            loss_imperceptibility = mse_criterion(embedded_image, images)
            
            # --- Attack & Inverse ---
            corrupted_image = apply_corruptions(embedded_image) 
            
            # Input to inverse: Corrupted Image + ZEROS
            attack_tensor = torch.cat([corrupted_image, torch.zeros_like(watermarks)], dim=1)
            
            recovered_full = model.inverse(attack_tensor)
            recovered_watermark = recovered_full[:, 3:, :, :]
            
            loss_extraction = mse_criterion(recovered_watermark, watermarks)
            
            # --- Optimization ---
            loss = (W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()

        # Update Learning Rate based on loss
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Logging
        with torch.no_grad():
            psnr = 10 * torch.log10(1 / loss_imperceptibility)
        
        print(f"Epoch: {epoch+1} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f}dB | Ext_MSE: {loss_extraction.item():.6f}")

        if ((epoch + 1) % 10 == 0):
            plot_for_one_img(model, test_image_path=TEST_IMG_PATH, device=device)
        # Save Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"revnet_checkpoint_{epoch+1}.pth")
            
def training_fn_refined_weighted_mse(EPOCHS, model, train_loader, device, criterion, optimizer):
    # 1. Use MSE for both.
    mse_criterion = criterion    
    
    weights = torch.tensor([1.0, 2.0, 0.5]).view(1, 3, 1, 1).to(device)

    def weighted_mse_loss(input, target, weights):
        diff = (input - target) ** 2
        weighted_diff = diff * weights
        return weighted_diff.mean() 
    
    # 3. Setup Optimizer and Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    fixed_watermark = generate_watermark_matrix(BATCH_SIZE, first_batch.size(2), first_batch.size(3)).to(device)
    
    print("--- Starting Extended Training ---")
    
    for epoch in range(EPOCHS):
        # Dynamic weighting
        # if epoch < 20:
        #     W_IMPERCEPTIBILITY = 1.0  # Let it learn the mechanism
        # elif epoch < 60:
        #     W_IMPERCEPTIBILITY = 10.0 # Force it to clean up
        # else:
        #     W_IMPERCEPTIBILITY = 50.0
        model.train()
        running_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            current_batch_size = images.size(0)
            
            if current_batch_size != BATCH_SIZE:
                watermarks = fixed_watermark[:current_batch_size]
            else:
                watermarks = fixed_watermark
            
            # --- Forward ---
            input_tensor = torch.cat([images, watermarks], dim=1)
            embedded_full = model(input_tensor)
            embedded_image = embedded_full[:, :3, :, :]
            
            #loss_imperceptibility = weighted_mse_loss(embedded_image, images, weights)
            loss_imperceptibility = mse_criterion(embedded_image, images)
            
            # --- Attack & Inverse ---
            corrupted_image = apply_corruptions(embedded_image) 
            
            # Input to inverse: Corrupted Image + ZEROS
            attack_tensor = torch.cat([corrupted_image, torch.zeros_like(watermarks)], dim=1)
            
            recovered_full = model.inverse(attack_tensor)
            recovered_watermark = recovered_full[:, 3:, :, :]
            
            loss_extraction = mse_criterion(recovered_watermark, watermarks)
            
            # --- Optimization ---
            loss = (W_IMPERCEPTIBILITY * loss_imperceptibility) + (W_EXTRACTION * loss_extraction)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()

        # Update Learning Rate based on loss
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Logging
        with torch.no_grad():
            psnr = 10 * torch.log10(1 / loss_imperceptibility)
        
        print(f"Epoch: {epoch+1} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f}dB | Ext_MSE: {loss_extraction.item():.6f}")

        if ((epoch + 1) % 10 == 0):
            plot_for_one_img(model, test_image_path=TEST_IMG_PATH, device=device)
        # Save Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"revnet_checkpoint_{epoch+1}.pth")