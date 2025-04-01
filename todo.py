# TODO:
#           1. Move into a different colourspace: Change axes from RGB to HSL
#                   (Hue, Saturation, Luminance)
#           2. Find the target skin tone: The skin tone is the Hue. Do a histogram
#                   binning the average of the skin tone?
#           3. Create a diff image: This shows how different each pixel is from the
#                   target skin tone. The green spill will have a much bigger difference
#           4. Create a binary mask/matt for green spill (green spill/not green spill)
#                   then ask to correct what is marked as binary green spill