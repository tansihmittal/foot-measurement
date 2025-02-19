import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, remove_small_objects
from skimage.measure import label
from PIL import Image
import io
import tempfile
import os

def find_contour_intersection(contour, point, direction, max_distance=500):
    """Find intersection of a ray with a contour"""
    for dist in range(1, max_distance):
        test_point = point + direction * dist
        test_point = tuple(map(int, test_point))
        result = cv2.pointPolygonTest(contour, test_point, False)
        if result < 0:  # Outside the contour
            return test_point
    return None

def detect_foot(binary_image):
    """Better foot detection with shape analysis and position heuristics."""
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Filter contours by aspect ratio and position
    for contour in contours:
        # Get bounding rectangle
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        
        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        
        # Typical foot has aspect ratio between 2.5 and 4
        if 2.5 <= aspect_ratio <= 4.0:
            # Calculate contour centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check if centroid is in the middle-bottom part of the image
                # (assuming foot is in the lower portion of image)
                h, w = binary_image.shape
                if cx > w*0.25 and cx < w*0.75 and cy > h*0.5:
                    return contour
    
    # If no suitable contour found, return the largest
    return contours[0] if contours else None

def calibrate_with_reference(image, reference_contour, reference_width_cm):
    """More accurate calibration using the reference object."""
    # Get the bounding box of the reference object
    ref_rect = cv2.minAreaRect(reference_contour)
    ref_box = cv2.boxPoints(ref_rect)
    ref_box = np.int0(ref_box)
    
    # Get actual width in pixels
    (_, _), (ref_width_px, ref_height_px), _ = ref_rect
    ref_dimension_px = max(ref_width_px, ref_height_px)
    
    # Calculate pixels per cm
    pixels_per_cm = ref_dimension_px / reference_width_cm
    return pixels_per_cm, ref_rect, ref_box

def advanced_foot_segmentation(image):
    """More accurate foot segmentation using multiple techniques."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply color-based segmentation in HSV space for skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Combine segmentation results
    combined_mask = cv2.bitwise_or(binary, skin_mask)
    
    # Clean with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def measure_foot_dimensions(foot_contour, pixels_per_cm):
    """More accurately measure foot dimensions along anatomical axes."""
    # Get rotated rectangle
    rect = cv2.minAreaRect(foot_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get dimensions and angle
    (x, y), (width, height), angle = rect
    
    # Make sure length is the longer dimension
    if width < height:
        width, height = height, width
        
    # Calculate the 'true' length by finding longest path through contour
    hull = cv2.convexHull(foot_contour)
    
    # Find the two points farthest from each other
    max_distance = 0
    farthest_points = None
    
    for i in range(len(hull)):
        for j in range(i+1, len(hull)):
            dist = np.sqrt((hull[i][0][0] - hull[j][0][0])**2 + 
                           (hull[i][0][1] - hull[j][0][1])**2)
            if dist > max_distance:
                max_distance = dist
                farthest_points = (hull[i][0], hull[j][0])
    
    # Calculate true foot length
    true_length_px = max_distance
    true_length_cm = true_length_px / pixels_per_cm
    
    # For width, use perpendicular to the length axis
    # We'll sample at 1/3 and 2/3 along the foot length and take maximum
    length_vector = np.array(farthest_points[1]) - np.array(farthest_points[0])
    length_vector = length_vector / np.linalg.norm(length_vector)
    
    perp_vector = np.array([-length_vector[1], length_vector[0]])
    
    # Sample points along the foot length
    sample_points = [
        farthest_points[0] + length_vector * true_length_px * 1/3,
        farthest_points[0] + length_vector * true_length_px * 1/2,
        farthest_points[0] + length_vector * true_length_px * 2/3,
    ]
    
    max_width_px = 0
    width_points = None
    
    # For each sample point, find intersection with contour along perpendicular
    for point in sample_points:
        point = point.astype(int)
        left_edge = find_contour_intersection(foot_contour, point, -perp_vector)
        right_edge = find_contour_intersection(foot_contour, point, perp_vector)
        
        if left_edge is not None and right_edge is not None:
            width = np.linalg.norm(np.array(left_edge) - np.array(right_edge))
            if width > max_width_px:
                max_width_px = width
                width_points = (left_edge, right_edge)
    
    if width_points is None:
        # Fallback to rotated rectangle width
        true_width_cm = height / pixels_per_cm
    else:
        true_width_cm = max_width_px / pixels_per_cm
    
    return true_length_cm, true_width_cm, farthest_points, width_points, box

def calculate_shoe_size(length_cm, width_cm):
    """More accurate shoe size calculation considering width."""
    # More accurate size charts based on mondopoint system
    # Length conversion
    eu_size = 31.3 + (length_cm - 19.5) * 1.54
    us_men_size = (length_cm / 0.254) - 24
    us_women_size = us_men_size + 1.5
    uk_size = us_men_size - 0.5
    
    # Width classification
    width_relative = width_cm / length_cm
    
    if width_relative < 0.35:
        width_category = "Narrow (A)"
    elif width_relative < 0.38:
        width_category = "Regular (B/C/D)"
    else:
        width_category = "Wide (E/EE)"
    
    # Account for brand variations (simplified)
    brand_adjustments = {
        'nike': {'adjustment': -0.5, 'width_note': 'Tends to run narrow'},
        'adidas': {'adjustment': 0, 'width_note': 'True to size'},
        'new_balance': {'adjustment': 0, 'width_note': 'Offers multiple width options'},
        'converse': {'adjustment': -1, 'width_note': 'Runs large and narrow'}
    }
    
    return {
        'estimated_eu_size': round(eu_size, 1),
        'estimated_us_men_size': round(us_men_size, 1),
        'estimated_us_women_size': round(us_women_size, 1),
        'estimated_uk_size': round(uk_size, 1),
        'width_category': width_category,
        'brand_adjustments': brand_adjustments
    }

def load_foot_detection_model():
    """
    Load a pre-trained machine learning model for foot segmentation.
    This is a placeholder - in reality, you would load a trained model.
    """
    # In a real implementation, you would load a trained model:
    # model = tf.keras.models.load_model('path/to/foot_segmentation_model')
    # return model
    
    # For this example, we'll return a dummy function
    def dummy_model(image):
        # This simulates what a real model would do - use our advanced segmentation instead
        return advanced_foot_segmentation(image)
    
    return dummy_model

def segment_foot_with_ml(image, model):
    """
    Use machine learning for more accurate foot segmentation.
    """
    # For now, we'll just use our advanced segmentation function as the model
    mask = model(image)
    
    # Post-process mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def enhanced_measure_foot_from_image(image, reference_object_width_cm=None):
    """
    Enhanced version of the foot measurement function with improved accuracy.
    """
    # Convert streamlit uploaded image to OpenCV format
    image_array = np.array(image)
    original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Resize image while maintaining aspect ratio
    height, width = original_image.shape[:2]
    max_dimension = 1000
    scale = min(max_dimension / width, max_dimension / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(original_image, (new_width, new_height))
    
    # Load ML model for segmentation
    model = load_foot_detection_model()
    foot_mask = segment_foot_with_ml(image, model)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        st.error("No foot contour detected. Please try again with better lighting or different background.")
        return None
    
    # Detect the foot using improved detection
    foot_contour = detect_foot(foot_mask)
    
    if foot_contour is None:
        st.error("Could not identify a foot in the image. Please ensure your foot is clearly visible.")
        return None
    
    # Drawing and visualization image
    display_image = image.copy()
    
    # Handle calibration
    if reference_object_width_cm and len(contours) > 1:
        # Find the reference object - assume it's the second largest contour
        contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
        ref_contour = contours_by_area[1] if len(contours_by_area) > 1 else None
        
        if ref_contour is not None:
            pixels_per_cm, ref_rect, ref_box = calibrate_with_reference(
                image, ref_contour, reference_object_width_cm)
            cv2.drawContours(display_image, [ref_box], 0, (0, 255, 255), 2)
            calibration_method = f"Using reference object: {pixels_per_cm:.2f} pixels/cm"
        else:
            # Fallback to statistical estimation
            # Use statistical average: typical male foot is ~26cm
            estimated_length_cm = 26
            pixels_per_cm = cv2.arcLength(foot_contour, True) / (estimated_length_cm * 2.2)  # perimeter heuristic
            calibration_method = f"Using statistical estimation: {pixels_per_cm:.2f} pixels/cm"
    else:
        # Use statistical average: typical male foot is ~26cm
        estimated_length_cm = 26
        pixels_per_cm = cv2.arcLength(foot_contour, True) / (estimated_length_cm * 2.2)  # perimeter heuristic
        calibration_method = f"Using statistical estimation: {pixels_per_cm:.2f} pixels/cm"
    
    # Measure foot dimensions
    length_cm, width_cm, length_points, width_points, box = measure_foot_dimensions(foot_contour, pixels_per_cm)
    
    # Get size recommendations
    size_results = calculate_shoe_size(length_cm, width_cm)
    
    # Draw measurements on the image
    # Draw rotated bounding box
    cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)
    
    # Draw length measurement line
    if length_points:
        cv2.line(display_image, 
                tuple(length_points[0]), 
                tuple(length_points[1]),
                (255, 0, 0), 2)
        midpoint = ((length_points[0][0] + length_points[1][0]) // 2,
                   (length_points[0][1] + length_points[1][1]) // 2)
        cv2.putText(display_image, f"{length_cm:.1f} cm", 
                   midpoint, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2)
    
    # Draw width measurement line
    if width_points:
        cv2.line(display_image, 
                tuple(width_points[0]), 
                tuple(width_points[1]),
                (0, 0, 255), 2)
        midpoint = ((width_points[0][0] + width_points[1][0]) // 2,
                   (width_points[0][1] + width_points[1][1]) // 2)
        cv2.putText(display_image, f"{width_cm:.1f} cm", 
                   midpoint, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(foot_mask, cmap='gray')
    axes[1].set_title('Foot Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Measurements\nLength: {length_cm:.1f} cm\nWidth: {width_cm:.1f} cm")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    viz_image = Image.open(buf)
    
    # Compile all results
    results = {
        'length_cm': round(length_cm, 1),
        'width_cm': round(width_cm, 1),
        'width_category': size_results['width_category'],
        'estimated_eu_size': size_results['estimated_eu_size'],
        'estimated_us_men_size': size_results['estimated_us_men_size'],
        'estimated_us_women_size': size_results['estimated_us_women_size'],
        'estimated_uk_size': size_results['estimated_uk_size'],
        'brand_adjustments': size_results['brand_adjustments'],
        'calibration_method': calibration_method,
        'visualization': viz_image
    }
    
    return results

def add_photo_taking_guidelines():
    st.markdown("""
    ### Tips for Taking Accurate Foot Measurements:

    1. **Proper Surface**: Place foot on a flat, contrasting background (white paper works well)
    2. **Lighting**: Ensure even lighting without shadows
    3. **Perspective**: Take photo directly from above (90° angle)
    4. **Reference**: Place reference object (credit card, ruler) at same level as foot
    5. **Foot Position**: Keep foot flat with toes extended naturally
    6. **Both Feet**: Measure both feet as they often differ in size
    """)

def main():
    st.set_page_config(page_title="Advanced Foot Measurement App", layout="wide")
    
    st.title("Advanced Shoe Size Recommendation System")
    st.markdown("""
    Upload a photo of your foot to get precise measurements and shoe size recommendations.
    Our enhanced algorithm uses advanced image analysis for more accurate results.
    """)
    
    # Add photo guidelines
    add_photo_taking_guidelines()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Photo")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        use_reference = st.checkbox("I have a reference object in the image")
        reference_width = None
        
        if use_reference:
            reference_width = st.number_input(
                "Enter the width of your reference object (cm):", 
                min_value=1.0, 
                max_value=50.0, 
                value=8.5,  # Credit card width as default
                step=0.5
            )
            st.info("Reference objects could be a credit card (8.5cm), ruler, or any object with known width. Place it at the same level as your foot.")
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Your Uploaded Image")
            st.image(image, use_column_width=True)
            
            if st.button("Measure Foot"):
                with st.spinner("Processing image with enhanced algorithms..."):
                    results = enhanced_measure_foot_from_image(
                        image, 
                        reference_object_width_cm=reference_width if use_reference else None
                    )
                
                if results:
                    with col2:
                        st.subheader("Enhanced Measurement Results")
                        st.info(results['calibration_method'])
                        
                        st.image(results['visualization'], use_column_width=True)
                        
                        st.subheader("Foot Dimensions")
                        st.markdown(f"**Length:** {results['length_cm']} cm")
                        st.markdown(f"**Width:** {results['width_cm']} cm")
                        st.markdown(f"**Width Category:** {results['width_category']}")
                        
                        st.subheader("Size Recommendations")
                        st.markdown(f"**EU Size:** {results['estimated_eu_size']}")
                        st.markdown(f"**US Men's Size:** {results['estimated_us_men_size']}")
                        st.markdown(f"**US Women's Size:** {results['estimated_us_women_size']}")
                        st.markdown(f"**UK Size:** {results['estimated_uk_size']}")
                        
                        st.warning("""
                        **Note:** These are estimates based on enhanced measurement algorithms.
                        Actual sizing may vary between brands. For the most accurate fit,
                        always refer to the specific brand's size chart.
                        """)
                        
                        # Add brand-specific advice
                        with st.expander("Brand-Specific Sizing Tips"):
                            for brand, info in results['brand_adjustments'].items():
                                st.markdown(f"- **{brand.capitalize()}:** {info['width_note']}. " + 
                                           (f"Consider going {abs(info['adjustment'])} size " + 
                                           ("up" if info['adjustment'] > 0 else "down") if info['adjustment'] != 0 else ""))
                        
                        # Share button
                        if st.button("Export Measurement Results"):
                            # Create a comprehensive report
                            report_buffer = io.BytesIO()
                            results['visualization'].save(report_buffer, format='PNG')
                            report_buffer.seek(0)
                            
                            # Provide download button
                            st.download_button(
                                label="Download Results as PNG",
                                data=report_buffer,
                                file_name="enhanced_foot_measurement_results.png",
                                mime="image/png"
                            )
    else:
        with col2:
            st.info("Please upload an image to get started!")
            st.image("https://www.wikihow.com/images/thumb/c/c3/Measure-Feet-for-Shoes-Step-5-Version-2.jpg/aid14166-v4-728px-Measure-Feet-for-Shoes-Step-5-Version-2.jpg", 
                     caption="Example of a good foot measurement photo")
            
            st.markdown("""
            ### Benefits of Our Enhanced Measurement:
            
            1. **Advanced Segmentation**: Better isolates foot from background
            2. **Anatomical Measurement**: Measures along true foot axis
            3. **Width Analysis**: Multiple sampling points for accurate width
            4. **Smart Calibration**: Uses reference objects or statistical estimation
            5. **Brand-Specific Sizing**: Accounts for variations between manufacturers
            """)

if __name__ == "__main__":
    main()
