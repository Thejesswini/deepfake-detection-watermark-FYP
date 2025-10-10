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
            