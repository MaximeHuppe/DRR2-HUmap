import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import imageio
from tqdm import tqdm
import glob
from natsort import natsorted

def create_gif_from_tiff_sequence(tiff_directory, output_path, duration=200, resize_factor=0.5, slice_step=1):
    """
    Create an animated GIF from a sequence of TIFF files.
    
    Args:
        tiff_directory: Path to directory containing TIFF files
        output_path: Output path for the GIF file
        duration: Duration between frames in milliseconds
        resize_factor: Factor to resize images (0.5 = half size for smaller file)
        slice_step: Step size for slices (1=all, 2=every 2nd, 5=every 5th, etc.)
    """
    print("Creating GIF from TIFF sequence...")
    
    # Get all TIFF files and sort them naturally
    tiff_files = glob.glob(os.path.join(tiff_directory, "*.tiff"))
    tiff_files = natsorted(tiff_files)  # Natural sorting for proper order
    
    if not tiff_files:
        print("No TIFF files found in directory!")
        return
    
    # Apply slice stepping
    if slice_step > 1:
        tiff_files = tiff_files[::slice_step]
        print(f"Using every {slice_step} slices: {len(tiff_files)} frames (from {len(glob.glob(os.path.join(tiff_directory, '*.tiff')))} total)")
    else:
        print(f"Found {len(tiff_files)} TIFF files")
    
    # Load and process images
    frames = []
    for tiff_file in tqdm(tiff_files, desc="Loading TIFF files"):
        # Load image
        img = Image.open(tiff_file)
        
        # Resize if requested
        if resize_factor != 1.0:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        frames.append(img)
    
    # Save as GIF
    print(f"Saving GIF with {len(frames)} frames...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,  # Infinite loop
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / (1024*1024)  # MB
    print(f"GIF saved: {output_path} ({file_size:.1f} MB)")

def create_mp4_from_tiff_sequence(tiff_directory, output_path, fps=10, resize_factor=0.5):
    """
    Create an MP4 video from a sequence of TIFF files using imageio.
    
    Args:
        tiff_directory: Path to directory containing TIFF files
        output_path: Output path for the MP4 file
        fps: Frames per second
        resize_factor: Factor to resize images
    """
    print("Creating MP4 from TIFF sequence...")
    
    # Get all TIFF files and sort them naturally
    tiff_files = glob.glob(os.path.join(tiff_directory, "*.tiff"))
    tiff_files = natsorted(tiff_files)
    
    if not tiff_files:
        print("No TIFF files found in directory!")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Create video writer
    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    
    try:
        for tiff_file in tqdm(tiff_files, desc="Processing frames"):
            # Load image
            img = Image.open(tiff_file)
            
            # Resize if requested
            if resize_factor != 1.0:
                new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB array
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            writer.append_data(img_array)
            
    finally:
        writer.close()
    
    file_size = os.path.getsize(output_path) / (1024*1024)  # MB
    print(f"MP4 saved: {output_path} ({file_size:.1f} MB)")

def create_matplotlib_animation(tiff_directory, output_path, fps=10, dpi=100):
    """
    Create animation using matplotlib (can save as GIF or MP4).
    
    Args:
        tiff_directory: Path to directory containing TIFF files
        output_path: Output path for animation
        fps: Frames per second
        dpi: Resolution for output
    """
    print("Creating matplotlib animation...")
    
    # Get all TIFF files and sort them naturally
    tiff_files = glob.glob(os.path.join(tiff_directory, "*.tiff"))
    tiff_files = natsorted(tiff_files)
    
    if not tiff_files:
        print("No TIFF files found in directory!")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Load first image to get dimensions
    first_img = Image.open(tiff_files[0])
    img_array = np.array(first_img)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Initialize image display
    im = ax.imshow(img_array, animated=True)
    
    # Animation function
    def animate(frame_num):
        if frame_num < len(tiff_files):
            img = Image.open(tiff_files[frame_num])
            img_array = np.array(img)
            im.set_array(img_array)
            
            # Extract slice number from filename
            slice_num = os.path.basename(tiff_files[frame_num]).split('_')[-1].split('.')[0]
            ax.set_title(f'Axial Slice {slice_num}', fontsize=16, fontweight='bold')
        
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(tiff_files), 
        interval=1000//fps, blit=True, repeat=True
    )
    
    # Save animation
    if output_path.endswith('.gif'):
        print("Saving as GIF...")
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    elif output_path.endswith('.mp4'):
        print("Saving as MP4...")
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)
    else:
        print("Unsupported format. Use .gif or .mp4")
        return
    
    plt.close()
    
    file_size = os.path.getsize(output_path) / (1024*1024)  # MB
    print(f"Animation saved: {output_path} ({file_size:.1f} MB)")

def create_interactive_viewer(tiff_directory):
    """
    Create an interactive matplotlib viewer with slider.
    
    Args:
        tiff_directory: Path to directory containing TIFF files
    """
    from matplotlib.widgets import Slider
    
    print("Creating interactive viewer...")
    
    # Get all TIFF files and sort them naturally
    tiff_files = glob.glob(os.path.join(tiff_directory, "*.tiff"))
    tiff_files = natsorted(tiff_files)
    
    if not tiff_files:
        print("No TIFF files found in directory!")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Load all images
    images = []
    print("Loading images for interactive viewing...")
    for tiff_file in tqdm(tiff_files):
        img = Image.open(tiff_file)
        images.append(np.array(img))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.1)
    
    # Display first image
    im = ax.imshow(images[0])
    ax.axis('off')
    ax.set_title(f'Axial Slice 0/{len(images)-1}', fontsize=16, fontweight='bold')
    
    # Create slider
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(images)-1, valinit=0, valfmt='%d')
    
    # Update function
    def update(val):
        idx = int(slider.val)
        im.set_array(images[idx])
        slice_num = os.path.basename(tiff_files[idx]).split('_')[-1].split('.')[0]
        ax.set_title(f'Axial Slice {slice_num} ({idx}/{len(images)-1})', 
                    fontsize=16, fontweight='bold')
        fig.canvas.draw()
    
    slider.on_changed(update)
    
    print("Interactive viewer ready! Use the slider to navigate through slices.")
    plt.show()

def main():
    """Main function with multiple animation options."""
    
    # Set your TIFF directory path
    tiff_directory = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/tiff_sequence_axial"
    output_dir = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2"
    
    print("🎬 TIFF Sequence Animation Creator")
    print("="*50)
    
    # Check if directory exists
    if not os.path.exists(tiff_directory):
        print(f"Directory not found: {tiff_directory}")
        return
    
    # Get total slice count for reference
    all_tiff_files = glob.glob(os.path.join(tiff_directory, "*.tiff"))
    total_slices = len(all_tiff_files)
    print(f"📊 Total slices available: {total_slices}")
    
    # LinkedIn-optimized GIFs with different slice steps
    print("\n🔥 Creating LinkedIn-optimized GIFs with different slice steps...")
    
    # Every 3rd slice (85 frames)
    linkedin_gif_3_path = os.path.join(output_dir, "ct_reconstruction_linkedin_every3.gif")
    create_gif_from_tiff_sequence(
        tiff_directory, 
        linkedin_gif_3_path, 
        duration=200,  # Slightly slower since fewer frames
        resize_factor=0.3,
        slice_step=3
    )
    
    # Every 5th slice (51 frames)
    linkedin_gif_5_path = os.path.join(output_dir, "ct_reconstruction_linkedin_every5.gif")
    create_gif_from_tiff_sequence(
        tiff_directory, 
        linkedin_gif_5_path, 
        duration=250,  # Slower since much fewer frames
        resize_factor=0.3,
        slice_step=5
    )
    
    # Every 10th slice (26 frames) - very compact
    linkedin_gif_10_path = os.path.join(output_dir, "ct_reconstruction_linkedin_every10.gif")
    create_gif_from_tiff_sequence(
        tiff_directory, 
        linkedin_gif_10_path, 
        duration=300,  # Even slower for good viewing
        resize_factor=0.3,
        slice_step=10
    )
    
    # Original full version (for comparison)
    print("\n📱 Creating original LinkedIn version (all slices)...")
    linkedin_gif_path = os.path.join(output_dir, "ct_reconstruction_linkedin_full.gif")
    create_gif_from_tiff_sequence(
        tiff_directory, 
        linkedin_gif_path, 
        duration=180,
        resize_factor=0.3,
        slice_step=1
    )
    
    print("\n✅ LinkedIn GIF creation complete!")
    print(f"📁 Files saved in: {output_dir}")
    print(f"📊 Slice comparison:")
    print(f"   • Every 3rd slice:  ~{total_slices//3} frames ({linkedin_gif_3_path.split('/')[-1]})")
    print(f"   • Every 5th slice:  ~{total_slices//5} frames ({linkedin_gif_5_path.split('/')[-1]})")
    print(f"   • Every 10th slice: ~{total_slices//10} frames ({linkedin_gif_10_path.split('/')[-1]})")
    print(f"   • Full version:     {total_slices} frames ({linkedin_gif_path.split('/')[-1]})")

if __name__ == "__main__":
    main() 