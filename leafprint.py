import cv2

# Load the image
img = cv2.imread('1.jpg')

# Display the original image
cv2.imshow('Original image', img)
cv2.waitKey(0)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale image', gray)
cv2.waitKey(0)

# Threshold the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Display the thresholded image
cv2.imshow('Thresholded image', thresh)
cv2.waitKey(0)

# Clean up the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Display the cleaned image
cv2.imshow('Cleaned image', clean)
cv2.waitKey(0)

# Find the contours
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the area
if len(contours) > 0:
    # Calculate the conversion factor
    scale_bar_length = 50 # in pixels
    scale_bar_real_length = 1 # in cm
    px_per_cm = scale_bar_length / scale_bar_real_length
    # Calculate the area in cm^2
    area_px = cv2.contourArea(contours[0])
    area_cm = area_px / (px_per_cm ** 2)
    print("Leaf area:",area_cm,"cm^2")
    # Draw the contour and the area
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.putText(img, f'Leaf area: {area_cm:.2f} cm^2', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the final image
    cv2.imshow('Final image', img)
    cv2.waitKey(0)
else:
    print('No leaf found')
