import os
import tempfile
import uuid
from functools import lru_cache

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import torch
import huggingface_hub
from diffusers import DiffusionPipeline
from transformers import pipeline
from potrace import Bitmap, POTRACE_TURNPOLICY_BLACK, POTRACE_TURNPOLICY_MINORITY

# Increase timeout for model downloads
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 900  # 15 minutes

# Define options
DIMENSIONAL_OPTIONS = ["2D", "3D"]

def create_prompt(user_prompt, is_cartoon, is_fourk, dim_option):
    """Create a combined prompt based on user selections."""
    options = []
    if is_cartoon:
        options.append("cartoon")
    if is_fourk:
        options.append("4k")
    if dim_option:
        options.append(dim_option.lower())
    return f"{user_prompt}, {', '.join(options)}" if options else user_prompt

@st.cache_resource
def load_diffusion_pipeline():
    """Load the diffusion pipeline."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            resume_download=True,
        ).to(device)
        return pipe
    except Exception as error:
        st.error(f"Error loading DiffusionPipeline: {str(error)}")
        return None

@st.cache_data
def generate_image(user_prompt, is_cartoon, is_fourk, dim_option, steps):
    """Generate an image based on the given parameters."""
    pipe = load_diffusion_pipeline()
    if pipe is None:
        return None

    combined_prompt = create_prompt(user_prompt, is_cartoon, is_fourk, dim_option)

    try:
        guidance_scale = 2.5
        result_image = pipe(
            prompt=combined_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]
        return result_image
    except Exception as error:
        st.error(f"Error generating image: {str(error)}")
        return None

@st.cache_resource
def load_rmbg_pipeline():
    """Load the background removal pipeline."""
    try:
        return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    except Exception as error:
        st.error(f"Error loading background removal model: {error}")
        return None

@st.cache_data
def remove_background(input_image):
    """Remove the background from the given image."""
    if not isinstance(input_image, Image.Image):
        raise TypeError("Input image must be a PIL image")

    rmbg_pipe = load_rmbg_pipeline()
    if rmbg_pipe is None:
        return input_image

    result = rmbg_pipe(input_image)

    if isinstance(result, Image.Image):
        return result
    if isinstance(result, list) and result and 'mask' in result[0]:
        return Image.fromarray(result[0]['mask'])
    
    raise ValueError("Unexpected result type from background removal pipeline")

@st.cache_data
def enhance_edges(input_image, edge_params=None):
    """Enhance the edges of the given image using Pillow."""
    if edge_params is None:
        edge_params = {
            'blur_radius': 2,
            'edge_enhance': 2,
        }
    
    # Convert to grayscale
    gray_image = input_image.convert("L")
    
    # Apply Gaussian blur
    blurred_image = gray_image.filter(ImageFilter.GaussianBlur(radius=edge_params['blur_radius']))
    
    # Find edges
    edges = blurred_image.filter(ImageFilter.FIND_EDGES)
    
    # Enhance edges
    for _ in range(edge_params['edge_enhance']):
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # Invert the image so edges are black on white background
    edges = ImageOps.invert(edges)
    
    return edges

def create_svg_path(curve):
    """Create SVG path from a curve."""
    parts = []
    start_point = curve.start_point
    parts.append(f"M{start_point.x},{start_point.y}")
    for segment in curve.segments:
        if segment.is_corner:
            control_point = segment.c
            end_point = segment.end_point
            parts.append(
                f"L{control_point.x},{control_point.y}L{end_point.x},{end_point.y}"
            )
        else:
            control_point1 = segment.c1
            control_point2 = segment.c2
            end_point = segment.end_point
            parts.append(
                f"C{control_point1.x},{control_point1.y} "
                f"{control_point2.x},{control_point2.y} "
                f"{end_point.x},{end_point.y}"
            )
    parts.append("z")
    return "".join(parts)

def file_to_svg(input_image, filename, output_dir, svg_fill_type):
    """Convert the image to SVG format."""
    edge_enhanced_image = enhance_edges(input_image)
    bitmap_data = np.array(edge_enhanced_image, dtype=np.uint8)
    bitmap = Bitmap(bitmap_data)
    path = bitmap.trace(
        turdsize=1,
        turnpolicy=POTRACE_TURNPOLICY_BLACK,
        alphamax=0.5,
        opticurve=False,
        opttolerance=0.1
    )

    output_path = os.path.join(output_dir, f"{filename}_{svg_fill_type}.svg")
    with open(output_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(
            f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
            f'width="{bitmap_data.shape[1]}" height="{bitmap_data.shape[0]}">'
        )
        for curve in path:
            svg_path = create_svg_path(curve)
            if svg_fill_type == "bw":
                svg_file.write(
                    f'<path d="{svg_path}" fill="none" '
                    f'stroke="black" stroke-width="1"/>'
                )
            elif svg_fill_type == "filled":
                svg_file.write(
                    f'<path d="{svg_path}" fill="black" '
                    f'stroke="none" stroke-width="1"/>'
                )
        svg_file.write("</svg>")
    return output_path

def file_to_svg_beta(image, filename):
    """Convert the image to SVG format using the beta version."""
    bitmap = Bitmap(np.array(image.convert("L")), blacklevel=0.5)
    plist = bitmap.trace(
        turdsize=2,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=1,
        opticurve=False,
        opttolerance=0.2,
    )

    output_path = os.path.join(tempfile.gettempdir(), f"{filename}_beta.svg")
    with open(output_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(
            f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{image.width}" height="{image.height}" '
            f'viewBox="0 0 {image.width} {image.height}">'
        )
        parts = []
        for curve in plist:
            start_point = curve.start_point
            parts.append(f"M{start_point.x},{start_point.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    control_point = segment.c
                    end_point = segment.end_point
                    parts.append(
                        f"L{control_point.x},{control_point.y}"
                        f"L{end_point.x},{end_point.y}"
                    )
                else:
                    control_point1 = segment.c1
                    control_point2 = segment.c2
                    end_point = segment.end_point
                    parts.append(
                        f"C{control_point1.x},{control_point1.y} "
                        f"{control_point2.x},{control_point2.y} "
                        f"{end_point.x},{end_point.y}"
                    )
            parts.append("z")
        svg_file.write(
            f'<path stroke="none" fill="black" '
            f'fill-rule="evenodd" d="{"".join(parts)}"/>'
        )
        svg_file.write("</svg>")
    return output_path

# Streamlit app
st.title("Image to SVG Converter")

# Initialize session state variables
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'bg_removed_image' not in st.session_state:
    st.session_state.bg_removed_image = None
if 'current_image_id' not in st.session_state:
    st.session_state.current_image_id = None

image_source = st.radio("Select image source:", ("Upload Image", "Generate Image"))

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Generate a new unique ID for this upload
        new_image_id = str(uuid.uuid4())
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.current_image_id = new_image_id
        st.session_state.bg_removed_image = None
        st.image(st.session_state.current_image, caption="Uploaded Image", use_column_width=True)

elif image_source == "Generate Image":
    prompt = st.text_input("Enter a prompt for image generation")
    cartoon = st.checkbox("Cartoon style")
    fourk = st.checkbox("4K resolution")
    dimensional_option = st.selectbox("Dimensional style", DIMENSIONAL_OPTIONS)
    num_inference_steps = st.slider("Number of inference steps", 1, 100, 50)
    
    if st.button("Generate Image"):
        generated_result = generate_image(
            prompt, cartoon, fourk, dimensional_option, num_inference_steps
        )
        if generated_result is not None:
            new_image_id = str(uuid.uuid4())
            st.session_state.current_image = generated_result
            st.session_state.current_image_id = new_image_id
            st.session_state.bg_removed_image = None
            st.image(generated_result, caption="Generated Image", use_column_width=True)

if st.session_state.current_image is not None:
    if st.button("Remove Background"):
        st.session_state.bg_removed_image = remove_background(st.session_state.current_image)
        st.image(st.session_state.bg_removed_image, caption="Image with Background Removed", use_column_width=True)

    if st.button("Show Enhanced Image"):
        image_to_enhance = st.session_state.bg_removed_image if st.session_state.bg_removed_image is not None else st.session_state.current_image
        edge_enhanced_result = enhance_edges(image_to_enhance)
        
        if st.session_state.bg_removed_image is not None:
            st.session_state.bg_removed_image = edge_enhanced_result
        else:
            st.session_state.current_image = edge_enhanced_result
        
        st.image(edge_enhanced_result, caption="Enhanced Image", use_column_width=True)

    # Add option to choose SVG style
    svg_style = st.radio("Select SVG Style", ("Black and White", "Filled", "Beta Version"))
    if st.button("Convert to SVG"):
        image_to_convert = st.session_state.bg_removed_image if st.session_state.bg_removed_image is not None else st.session_state.current_image
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as tmp_file:
            svg_filename = os.path.basename(tmp_file.name)

            if svg_style == "Beta Version":
                svg_output_path = file_to_svg_beta(image_to_convert, svg_filename)
            else:
                SVG_FILL_TYPE = "bw" if svg_style == "Black and White" else "filled"
                svg_output_path = file_to_svg(
                    image_to_convert,
                    svg_filename,
                    os.path.dirname(tmp_file.name),
                    SVG_FILL_TYPE
                )

        with open(svg_output_path, "r", encoding="utf-8") as file:
            svg_content = file.read()

        st.download_button(
            label="Download SVG",
            data=svg_content,
            file_name=f"{svg_filename}_{svg_style.lower().replace(' ', '_')}.svg",
            mime="image/svg+xml"
        )

        os.unlink(svg_output_path)

        st.success("SVG conversion complete. Click the 'Download SVG' button to save the file.")

    # Display the current state of the image
    st.subheader("Current Image State")
    if st.session_state.bg_removed_image is not None:
        st.image(st.session_state.bg_removed_image, caption="Current Image (Background Removed)", use_column_width=True)
    else:
        st.image(st.session_state.current_image, caption="Current Image", use_column_width=True)

# Display current image_id for debugging
st.sidebar.text(f"Current Image ID: {st.session_state.current_image_id}")
